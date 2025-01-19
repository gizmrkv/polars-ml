# 1

## 仮説
year x countryごとのGDPとnum_soldの相関はある

## 検証方法
year x country x store x productごとにnum_soldとGDPの平均をとる．
num_sold vs GDPの散布図を見る．　

## 結果
store x productごとに分けるとnum_soldとGDPに強い正の相関がある．
切片はいずれも0に近い．傾きのみが異なる．

## 考察
store x productごとにintercept=Falseで線形回帰すれば，num_soldの年平均をGDPから予測できる．

store x productごとに
num_sold_adj = num_sold - GDP * coef
とすると，2011年，2015年の急変化が消えた．トレンドが消えた．
とてもよい．このあたりで次の実験に移るのが良いだろう．

しかし，num_soldに欠損値を持つstore x productはどう線形回帰すればよいか．
store, productごとに回帰曲線の傾きに与える影響を見て，欠損値を持つグループの傾きを手動で決めるとよいだろうか．

# 2

## 仮説
store x productごとの傾きには一定の法則がある

## 検証方法
store, productをOnehotEncodingして，線形回帰で傾きを予測する．
interceptの有無でそれぞれ求めて，差がなければintercept=Falseで回帰する．

## 結果
Mean squared error: 6.699453964361088e-06
R2 score: 0.9301989522385535
結構強い正の相関がある

## 考察
store x productごとの傾きは一定の法則があるといえる．
とはいえ多少のずれはある．

