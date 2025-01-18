# Forecasting Sticker Sales

## Competition
- 課題: 異なる国でのステッカー販売数の予測
- 評価指標: Mean Absolute Percentage Error (MAPE)
- 期間: 2025/1/1 - 2025/1/31
- リンク: [https://www.kaggle.com/competitions/playground-series-s5e1]


## Data
- train: (230_130, 6)
- test: (98_550, 5)
- columns:
    - `id`: int
    - `date`: str, YYYY-MM-DD
    - `country`: str
    - `store`: str
    - `product`: str
    - `num_sold`: float

## Analysis
- `date`
    - train: from 2010-01-01 to 2016-12-31
    - test: from 2017-01-01 to 2019-12-31
- `country`
    - ['Italy', 'Singapore', 'Norway', 'Finland', 'Kenya', 'Canada']
- `store`
    - ['Stickers for Less', 'Premium Sticker Mart', 'Discount Stickers']
- `product`
    - ['Kaggle', 'Holographic Goose', 'Kerneler Dark Mode', 'Kerneler', 'Kaggle Tiers']
- `num_sold`
    - 欠損値が8871個ある
        - Canada, Kenyaのみ欠損値がある
        - Holographic Gooseは最も欠損値が多い
        - shape: (9, 4)
            ┌─────────┬──────────────────────┬────────────────────┬──────────┐
            │ country ┆ store                ┆ product            ┆ num_sold │
            │ ---     ┆ ---                  ┆ ---                ┆ ---      │
            │ str     ┆ str                  ┆ str                ┆ u32      │
            ╞═════════╪══════════════════════╪════════════════════╪══════════╡
            │ Kenya   ┆ Discount Stickers    ┆ Kerneler           ┆ 63       │
            │ Canada  ┆ Discount Stickers    ┆ Kerneler           ┆ 1        │
            │ Kenya   ┆ Discount Stickers    ┆ Holographic Goose  ┆ 2557     │
            │ Kenya   ┆ Stickers for Less    ┆ Holographic Goose  ┆ 1358     │
            │ Canada  ┆ Discount Stickers    ┆ Holographic Goose  ┆ 2557     │
            │ Canada  ┆ Stickers for Less    ┆ Holographic Goose  ┆ 1308     │
            │ Kenya   ┆ Discount Stickers    ┆ Kerneler Dark Mode ┆ 1        │
            │ Kenya   ┆ Premium Sticker Mart ┆ Holographic Goose  ┆ 646      │
            │ Canada  ┆ Premium Sticker Mart ┆ Holographic Goose  ┆ 380      │
            └─────────┴──────────────────────┴────────────────────┴──────────┘
    - num_soldの平均は
        - Kenay << Italy < Finland < Canada = Singapore < Norway
        - HG <<> Ker < KDM < KT< K
        - Discount < Less < Premium
    - 2011年で急上昇
    - 2015年で急減少
    - 1日単位での変動が大きい（土日の影響？）
    - すべてのカテゴリ，4~6月ごろの変動が激しい
    - 年の変わり目でピークがある
    - Ker, KDMのみ年単位の周期がある
    - K, KT(, HGも？)は偶数年単位の周期がある
