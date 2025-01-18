n_unique
    country: 6
    store: 3
    product: 5

国，店，商品の組み合わせは90通り．
つまり，全通り現れている．

dateは日付，抜けはない．2556日分のデータがある．
2010-01-01から2016-12-31までのデータ．7年分
testは2017-01-01から2019-12-31までのデータ．3年分

num_soldにのみ欠損値がある．
TODO: 欠損の原因を調査する．

以下のnum_soldはすべて欠損している．
- "Canada"	"Discount Stickers"	"Holographic Goose"
- "Kenya"	"Discount Stickers"	"Holographic Goose"

欠損値のパターンで以下のように分類できる
- すべての日付で欠損している
- 一部の日付で欠損している
- すべての日付で欠損していない

国 x 店 x 商品ごとの売り上げを見る
- 年ごとに周期を持つ国 x 店 x 商品
- 年ごとに挙動が変わる国 x 店 x 商品
