# crypto-transformers
Repositório do 4º Desafio de dados (Dataton) de moedas digitais FGV EESP, by team HAS
## Instalação de dependências
Todos os scripts rodaram com Python 3.10.12 com os seguintes pacotes adicionais: numpy, pandas, scikit-learn e tensorflow

Para instalar cada um, execute em seu ambiente substituindo 'package_name' pelo nome do pacote:
```bash
pip install package_name
```
## Base de dados
Dados extraídos do banco do WHALE ALERT ([link](https://whale-alert.io/sample-data/)), mas vale ressaltar que esses dados são constantemente atualizados.  
Para reproduzir os resultados de nossos experimentos, baixe a exata versão que utilizamos em nosso [Google Drive](https://drive.google.com/drive/folders/1OepjThUsXGqMUAGwja9RyOq2XBKhTYQp).  
Coloque os datasets na pasta 'data' do repositório.

## Executando os experimentos
Para pré-processar os dados, execute 'preprocess_data.py' com o seguinte comando ou equivalente:
```bash
python3 preprocess_data.py
```
Para rodar o Transformer, execute 'Transformer.py' com o seguinte comando ou equivalente:
```bash
python3 Transformer.py
```
Os resultados serão mostrados na saída padrão, e podem demorar alguns minutos.
Os logs de nossos experimentos já estão na pasta 'results'.
