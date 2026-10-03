# References and background

Public sources consulted while setting up the project. Korean-language sources are summarized in English; quotations
are paraphrased unless marked. Items marked *checked* were opened and the cited content was confirmed on 2026-10-03;
the rest were read at the level of news reports or abstracts.

## 1. Market surveillance in Korea

| Topic | Summary | Source |
|---|---|---|
| Surveillance procedure | Example workflow in the financial regulators' guideline for continuous monitoring of abnormal trading on virtual-asset exchanges: detect suspicious instruments → review report → select suspect account groups → analyze trades per account → report → notify regulators or investigators | Financial Services Commission press release: https://www.fsc.go.kr/no010101/82943 |
| Virtual Asset User Protection Act | In force since 2024-07-19; exchanges must monitor abnormal trading continuously and report suspected unfair trading | https://www.fsc.go.kr/no010101/82682 |
| FSS moving-interval grid search | The Financial Supervisory Service (FSS) announced an algorithm that splits a suspect's trading period into sub-intervals and scans all of them at one-second resolution, processing hundreds of thousands of intervals in parallel on GPUs (2026-02) | Herald Economy: https://biz.heraldcorp.com/article/10667723 |
| FSS VISTA platform | Integrates order-book and trade data from five domestic exchanges and major foreign exchanges; groups accounts with similar trading behaviour (order timing, order channel) automatically; 24-hour monitoring (2026-05) | Money Today: https://www.mt.co.kr/stock/2026/05/03/2026043021225158509 |
| FSS AI investigation model | Planned for 2028: detects suspicious trades and structures relations among accounts | Seoul Economic Daily: https://www.sedaily.com/article/20084957 |
| KRX AI surveillance | Korea Exchange AI-based surveillance (2018) focused on accounts and linked accounts; detection time reported to drop from over a month to about an hour | Edaily: https://edaily.co.kr/News/Read?mediaCodeNo=257&newsId=03142246619081000 |
| KRX longer windows | Detection criteria extended from at most 100 days to 6-month and 1-year windows (2023) | Newsis: https://mobile.newsis.com/view.html?ar_id=NISX20230925_0002463132 |
| KRX person-level surveillance | Since 2025-10-28 accounts belonging to the same person are linked using pseudonymized data; the number of surveillance targets was expected to fall by about 39% | Korea Financial Newspaper: https://www.fntimes.com/html/view.php?ud=202510221945458573179ad43907_18 |
| Member-firm monitoring | All securities firms run first-line monitoring on the same exchange-defined criteria, with warnings and order refusal (2022 reform) | Edaily: https://edaily.co.kr/News/Read?mediaCodeNo=257&newsId=03056966632196736 |

## 2. Exchange alert rules (Upbit, *checked*)

Upbit help center, "Upbit market alert system" (updated 2026-09-30): https://support.upbit.com/hc/ko/articles/900005994766

| Alert | Computed | Announced | Compared against | Caution / Warning / Danger |
|---|---|---|---|---|
| Global price gap | every minute | on occurrence | CoinMarketCap price | ±10-20% / ±20-30% / ≥ ±30% |
| Price surge or drop | every minute | on occurrence | close 24 h earlier | rise 50-100%, fall 50-75% / rise 100-200%, fall 75-90% / rise ≥ 200%, fall ≥ 90% |
| Volume surge | daily 09:00 KST | 10:00 | last 24 h vs previous 3-day average | +300-400% / +400-500% / ≥ +500% |
| Deposit surge | daily 09:00 KST | 10:00 | same | same |
| Concentration in few accounts | daily 09:00 KST | 10:00 | buy or sell participation of top accounts over 24 h | 50-75% / 75-90% / ≥ 90% |

- The price rule changed on 2026-05-18 from "yesterday's close" to "close 24 h earlier".
- The page states that exact exception conditions are not disclosed, to prevent misuse and evasion of the alerts.
- The number of "top accounts" is not disclosed.
- Real-time daily-change rules on Korean equity markets work the same way (Korea Exchange volatility interruption:
  dynamic ±3% / ±6% from the last trade, static ±10% from the last single-price auction):
  https://www.newsis.com/view/NISX20220519_0001878246. The U.S. Limit Up-Limit Down mechanism uses a five-minute average
  reference price republished every 30 seconds: https://www.nasdaqtrader.com/content/MarketRegulation/LULD_FAQ.pdf

## 3. Enforcement cases (reported)

| Case | Reported pattern | Source |
|---|---|---|
| First conviction under the Virtual Asset User Protection Act (conduct 2024-07 to 2024-10, ruling 2026-02) | Automated high-price buying and low-price selling in quick succession; buy orders unlikely to execute, building an apparent buy wall | https://www.hankyung.com/article/2026020443831 |
| FSC decision (2025-11) | Two types: price pulled toward a target with pre-placed sell orders; small market orders repeated "several times per second for tens of minutes" through APIs | https://www.fsc.go.kr/no010101/85607 |
| FSC referral (2026-03) | Large buy orders at the moment the daily change rate resets, then selling within three minutes | https://view.asiae.co.kr/article/2026031819115878772 |
| FSC referral (2026-04) | Pre-buying then concentrated high-price buying; separately, API keys of many accounts borrowed to trade among those accounts | https://view.asiae.co.kr/article/2026042916215127302 |
| Hyperliquid JELLY (2025-03-26) | Price of a low-liquidity token pushed up across venues; the exchange vault absorbed losses; the contract was delisted | https://sg.finance.yahoo.com/news/hyperliquid-delists-jelly-vault-squeezed-160020190.html |
| Hyperliquid POPCAT (2025-11-12) | Funds split across 19 wallets for long positions, a large buy order posted and withdrawn, about $4.9M loss to the exchange | https://www.coindesk.com/markets/2025/11/13/peak-degen-warfare-alleged-popcat-manipulation-hits-hyperliquid-with-usd4-9m-loss |

## 4. Surveillance outside Korea

| Organization | Summary | Source |
|---|---|---|
| SEC MIDAS | Over one billion records per day from each U.S. equity exchange, microsecond timestamps | https://www.sec.gov/securities-topics/market-structure-analytics/midas-market-information-data-analytics-system |
| FINRA | Tens of billions of market events per day on average, hundreds of surveillance patterns | https://www.finra.org/about/how-we-operate/technology |
| Nasdaq | Deep-learning surveillance patterns (2019); its U.S. surveillance team reviews over 750,000 alerts per year | https://www.nasdaq.com/press-release/nasdaq-launches-artificial-intelligence-for-surveillance-patterns-on-u.s.-stock |
| CFTC | Adopted Nasdaq surveillance technology (2025) | https://www.cftc.gov/PressRoom/PressReleases/9110-25 |
| FINRA Rule 3110 | Broker-dealers must review transactions for manipulative trading | https://www.finra.org/rules-guidance/guidance/reports/2023-finras-examination-and-risk-monitoring-program/manipulative-trading |

## 5. Data

| Dataset | Notes | Source |
|---|---|---|
| Binance public archive (*checked*) | Klines and aggregated trades (`agg_id, price, qty, first_trade_id, last_trade_id, time, is_buyer_maker, is_best_match`); no account identifiers | https://data.binance.vision |
| Pump-and-dump event list (*checked*) | La Morgia et al.; 1,110 events from Telegram groups, 520 on Binance (85 symbols, 2018-01 to 2021-01, minute-level announcement times), MIT license | https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset |
| Hyperliquid (*checked*) | Public API trades carry both counterparties' wallet addresses; historical fills and order-book snapshots in requester-pays S3 buckets | https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data |

## 6. Related work

| Area | Work | Source |
|---|---|---|
| Synchronized-account detection | SynchroTrap, Cao et al., CCS 2014 (*checked*): clusters accounts whose actions match on the same object within a time window; partitions by object and time to scale | https://users.cs.duke.edu/~xwy/publications/SynchroTrap-ccs14.pdf |
| | CopyCatch, Beutel et al., WWW 2013 (pp. 119-130): lockstep behaviour; follow-up CatchSync, KDD 2014 | https://dl.acm.org/doi/10.1145/2623330.2623632 (CatchSync) |
| | FRAUDAR, Hooi et al., KDD 2016: dense-subgraph detection resistant to camouflage | https://bhooi.github.io/projects/fraudar/index.html |
| Collusion in markets | Wang, Zhou, Guan 2011: correlation of signed order flow between accounts on the Shanghai Futures Exchange, deployed as a surveillance tool | https://arxiv.org/abs/1110.1522 |
| | Palshikar and Apte 2008: collusion sets via graph clustering | https://www.semanticscholar.org/paper/Collusion-set-detection-using-graph-clustering-Palshikar-Apte/9a573e5b5a71d1110d93e752c898a303bcbfb564 |
| | Victor and Weintraud, WWW 2021: wash trading on decentralized exchanges | https://arxiv.org/pdf/2102.07001 |
| Coordinated wallets | Kamat 2026 (*checked*): persistent wallet cohorts on Pump.fun; outcome contamination in effect estimates | https://arxiv.org/pdf/2607.02795 |
| Spontaneous synchrony | Saavedra, Hagerty, Uzzi, PNAS 2011: synchronous trading among day traders arises without coordination | https://arxiv.org/abs/1110.0381 |
| Sliding-window aggregation | DABA (DEBS 2017), FiBA (PVLDB 2019), Scotty (TODS 2021) | https://hirzels.com/martin/papers/debs17-daba.pdf · https://www.vldb.org/pvldb/vol12/p1167-tangwongsan.pdf · https://dl.acm.org/doi/abs/10.1145/3433675 |
| Stream processing on GPUs | SABER (SIGMOD 2016), LightSaber (SIGMOD 2020) | https://www2.informatik.hu-berlin.de/~weidlima/pubs/koliousis_sigmod_2016_saber.pdf · https://lsds.doc.ic.ac.uk/sites/default/files/lightsaber-sigmod20.pdf |
| Pump-and-dump detection | La Morgia et al., ACM TOIT | https://dl.acm.org/doi/fullHtml/10.1145/3561300 |
| Scan statistics | Review of scan statistics | https://projecteuclid.org/journals/statistics-surveys/volume-15/issue-none/An-up-to-date-review-of-scan-statistics/10.1214/21-SS132.pdf |
| Matrix profile on GPUs | SCAMP, Zimmerman et al. | https://www.cs.ucr.edu/~eamonn/public/GPU_Matrix_profile_VLDB_30DraftOnly.pdf |

## 7. Events used in the pilot

| Event (UTC) | Sources |
|---|---|
| 2024-01-03: Matrixport note expects the SEC to reject spot ETFs; BTC drops | [crypto.news](https://crypto.news/sec-will-reject-all-bitcoin-etfs-in-january-says-matrixport/) · [Securities Docket](https://www.securitiesdocket.com/2024/01/03/matrixport-sec-to-reject-spot-bitcoin-etf-proposals-in-january/) |
| 2024-01-09 21:11: SEC X account compromised, fake spot-ETF approval post | [BleepingComputer](https://www.bleepingcomputer.com/news/security/us-secs-x-account-hacked-to-announce-fake-bitcoin-etf-approval/) · [CBS News](https://www.cbsnews.com/news/sec-hack-spot-bitcoin-etf-twitter-announcement-gary-gensler/) · [CNBC](https://www.cnbc.com/2024/01/09/sec-says-it-did-not-yet-approve-bitcoin-etf.html) |
| 2024-01-11 14:30: U.S. spot bitcoin ETFs start trading | [Morningstar](https://www.morningstar.com/funds/spot-bitcoin-etfs-trading-debut-6-charts) · [Grayscale Research](https://research.grayscale.com/market-commentary/january-2024-the-debut-of-spot-bitcoin-etfs) |
| 2024-02-28: BTC approaches $64k; Coinbase shows zero balances under load | [CoinDesk](https://www.coindesk.com/business/2024/02/28/coinbase-account-balances-shows-0-for-users-post-bitcoin-60k-breakout) · [CNBC](https://www.cnbc.com/2024/02/28/coinbase-users-see-0-balance-after-crypto-trading-app-suffers-glitch.html) |
| 2024-03-05: BTC record above $69k, then a drop of up to 10% | [CNBC](https://www.cnbc.com/2024/03/05/crypto-market-today.html) · [CoinDesk](https://www.coindesk.com/markets/2024/03/05/bitcoin-hit-a-record-high-heres-what-might-happen-next) |
| 2024 Q1 U.S. CPI releases (01-11, 02-13, 03-12) | [BLS CPI release schedule](https://www.bls.gov/schedule/news_release/cpi.htm) |
| 2024 Q1 FOMC statements (01-31, 03-20) | [Federal Reserve FOMC calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) |
