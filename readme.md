## Atividade G2 - Aprendizado de Máquina (Modelo + API)

<b>Nomes: </b>Flamarion Fagundes Pinto, Manuela Kleinkauf de Oliveira e Luan Sananda Rosa Custódio

API para classificação de resíduos sólidos (lata, garrafa PET, plástico, etc.) usando Python + FastAPI e um modelo de Machine Learning treinado durante o build do Docker.

O dataset utilizado foi o <b>Recyclable and Household Waste Classification</b> que pode ser baixado em:

https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification?resource=download

#### Executar via Docker

1. Coloque suas imagens no diretório `dataset/`

2. Rode:

```bash
docker compose up --build -d
```

A imagem Docker é responsável por criar um ambiente Python que irá treinar o modelo (durante o build) e levantar uma API que utiliza esse modelo para fazer predições baseado em novas imagens.

Para apagar o container e os volumes, basta rodar:

```bash
docker compose down -v
```

#### Endpoints

A documentação Swagger se encontra disponível em:

http://localhost:8000/docs

OBS: A mesma é gerada de maneira automática com FastAPI

#### Testar endpoints com cURL

- Chamada para o endpoint que retorna a classificação, utilizando a foto de uma garrafa de refrigerante

```bash
curl -X POST "http://localhost:8000/predict" \
 -H "accept: application/json" \
 -H "Content-Type: multipart/form-data" \
 -F "file=@./examples/garrafa-pet.jpg"
```

Resultado esperado:

```json
{
  "predicted_class": "plastic_soda_bottles",
  "predicted_class_pt": "Garrafas PET",
  "confidence": 1.0
}
```

- Chamada para o endpoint que retorna as classes

```bash
curl -X GET "http://localhost:8000/classes" \
  -H "accept: application/json"
```

Resultado esperado:

```json
{
  "classes": [
    {
      "id": "aerosol_cans",
      "label": "Latas de aerossol"
    },
    {
      "id": "aluminum_food_cans",
      "label": "Latas de alimentos (alumínio)"
    },
    ...
  ]
}
```
