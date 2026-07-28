# -*- coding: utf-8 -*-
"""KOSPI 일별 종가 차트 (1999.12 ~ 2001.9), 닷컴 버블 붕괴 구간.

데이터: pykrx (KRX 지수 1001), 실패 시 yfinance ^KS11 폴백.
출력: kospi_dotcom.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import pandas as pd

START, END = "19991201", "20010930"
CRASH_DAY = pd.Timestamp("2000-04-17")


def setup_korean_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    else:
        print("경고: 한글 폰트를 찾지 못했습니다. 나눔고딕 설치가 필요합니다.")
    plt.rcParams["axes.unicode_minus"] = False


def load_data():
    try:
        from pykrx import stock
        df = stock.get_index_ohlcv(START, END, "1001")
        if df.empty:
            raise RuntimeError("pykrx가 빈 데이터를 반환")
        close = df["종가"].astype(float)
        close.index = pd.to_datetime(close.index)
        return close.sort_index(), "pykrx (KRX 지수 1001)"
    except Exception as e:
        print(f"pykrx 실패 ({e!r}) → yfinance ^KS11로 폴백합니다.")
        import os
        import requests
        import yfinance as yf
        # 프록시 환경에서 curl_cffi TLS가 차단되는 경우가 있어 requests 세션을 사용
        session = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        ca = os.environ.get("REQUESTS_CA_BUNDLE") or "/root/.ccr/ca-bundle.crt"
        if os.path.exists(ca):
            session.verify = ca
        df = yf.download("^KS11", start="1999-12-01", end="2001-10-01",
                         progress=False, auto_adjust=False, session=session)
        if df.empty:
            raise RuntimeError("yfinance도 데이터를 가져오지 못했습니다.")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # MultiIndex 컬럼 대응
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.dropna().astype(float).sort_index(), "yfinance (^KS11)"


def plot(close, source):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    ax.plot(close.index, close.values, color="#4269d0", linewidth=1.4)

    ax.axvline(CRASH_DAY, color="red", linewidth=1.2, linestyle="--", alpha=0.9)
    crash_close = close.loc[CRASH_DAY] if CRASH_DAY in close.index else None
    y_annot = crash_close if crash_close is not None else close.median()
    ax.annotate(
        "2000.4.17 -11.6% 폭락",
        xy=(CRASH_DAY, y_annot),
        xytext=(35, 40), textcoords="offset points",
        fontsize=11, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
    )

    ax.set_title("KOSPI 일별 종가 (1999.12 ~ 2001.9)", fontsize=14, pad=12)
    ax.set_xlabel("날짜")
    ax.set_ylabel("지수 (pt)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(x=0.01)
    fig.text(0.99, 0.005, f"출처: {source}", ha="right", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig("kospi_dotcom.png", dpi=150)
    print("저장 완료: kospi_dotcom.png (150dpi)")


def verify(close):
    print("\n===== 데이터 검증 =====")
    print(f"거래일 수      : {len(close)}일")
    print(f"시작           : {close.index[0].date()}  종가 {close.iloc[0]:,.2f} pt")
    print(f"끝             : {close.index[-1].date()}  종가 {close.iloc[-1]:,.2f} pt")
    print(f"기간 고점      : {close.idxmax().date()}  {close.max():,.2f} pt")
    print(f"기간 저점      : {close.idxmin().date()}  {close.min():,.2f} pt")

    if CRASH_DAY in close.index:
        pos = close.index.get_loc(CRASH_DAY)
        day_close = close.iloc[pos]
        line = f"2000-04-17 종가: {day_close:,.2f} pt"
        if pos > 0:
            prev = close.iloc[pos - 1]
            chg = (day_close / prev - 1) * 100
            line += f" (전일 {close.index[pos-1].date()} {prev:,.2f} 대비 {chg:+.2f}%)"
        print(line)
    else:
        print("경고: 2000-04-17 데이터가 없습니다.")


def main():
    setup_korean_font()
    close, source = load_data()
    print(f"데이터 소스: {source}")
    plot(close, source)
    verify(close)


if __name__ == "__main__":
    sys.exit(main())
