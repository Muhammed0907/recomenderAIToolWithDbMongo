import faiss,requests,time
import numpy as np

OllamURL = "http://localhost:11434/api/embeddings"
OllamaEmbedModel = "nomic-embed-text"
training_status = {"is_training": False, "success": False, "current_instance": 0, "total_instances": 0}

def Request2Ollama(prompt:str):
    res = requests.post(
        OllamURL,
        json={
            "model":OllamaEmbedModel,
            "prompt":prompt,
        }
    )
    return res


def AddNewAIToolInModel(modelPath, newTextualData):
    trainedAitoolEmbedModel = faiss.read_index(modelPath)

    initial_count = trainedAitoolEmbedModel.ntotal
    print(f"initial_count of vectors in the model: {initial_count}")

    dim = 768  
    new_embeddings = np.zeros((len(newTextualData), dim), dtype="float32")
    
    for i, prompt in enumerate(newTextualData):
        try:
            res = requests.post(
                url=OllamURL,
                json={"model": OllamaEmbedModel, "prompt": prompt}
            )
            embedding = res.json()["embedding"]
        except:
            time.sleep(2)
            res = requests.post(
                url=OllamURL,
                json={"model": OllamaEmbedModel, "prompt": prompt}
            )
            embedding = res.json()["embedding"]
        
        new_embeddings[i] = np.array(embedding)

    trainedAitoolEmbedModel.add(new_embeddings)
    
    new_count = trainedAitoolEmbedModel.ntotal
    print(f"New count of vectors in the model: {new_count}")
    
    faiss.write_index(trainedAitoolEmbedModel, modelPath)
    print("Model updated and saved.")


def TrainModel(textualRepData,outputName):
    training_status["is_training"] = True
    training_status["success"] = False
    training_status["current_instance"] = 0
    training_status["total_instances"] = len(textualRepData)

    dim = 768
    index = faiss.IndexFlatL2(dim)
    x = np.zeros((len(textualRepData), dim), dtype="float32")

    for i, repres in enumerate(textualRepData):
        training_status["current_instance"] = i + 1
        try:
            res = requests.post(
                url=OllamURL,
                json={"model": OllamaEmbedModel, "prompt": repres}
            )
            embedding = res.json()["embedding"]
        except:
            time.sleep(2)
            res = requests.post(
                url=OllamURL,
                json={"model": OllamaEmbedModel, "prompt": repres}
            )
            embedding = res.json()["embedding"]

        x[i] = np.array(embedding)

    index.add(x)
    training_status["is_training"] = False
    training_status["success"] = True
    faiss.write_index(index, outputName)
