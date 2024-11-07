## ファイルについて説明

- drug_bankv1

    - DrugBankのサイトから薬剤として承認されている化合物を取得(https://go.drugbank.com/releases/5-1-10)
    - ```drug_bankv1/drug_bank_structures_filtered.sdf```：取得した化合物
    - ```drug_bankv1/drug_bank_fragment_filtered.sdf```：化合物を分割した後のフラグメント種
    - ```drug_bankv1/drug_bank_output_filtered.sdf```：化合物がどのフラグメントから構成されているのかの情報を持つ


- ```auto_clustering_new.ipynb```
    - 研究の実験のコードが全て記載
    - 詳しくはコメントを確認
- ```Function.py```
    - ```auto_clustering_new.ipynb```で利用する関数の定義



## 使い方
```
    auto_clustering_new
    ├── drug_bankv1
    ├── auto_clustering_new.ipynb
    └── Function
         └── ...
```
のファイルをグーグルコラボにダウンロードして実行
- filepath　　という変数を定義してるので、それだけ自分のディレクトリ構成に合わせて変更する
