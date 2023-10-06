from fastapi import FastAPI
import trainCustomModel, getUserRecommendation

app = FastAPI()

@app.get("/train")
def read_root():
    trainCustomModel.model()
    return trainCustomModel.model()

@app.get("/recommend/")
def read_item(query_param: str = None):
    if query_param != None:
        print("query_param", query_param)
        return getUserRecommendation.recommend(query_param)
    else:
        return "Query is None"