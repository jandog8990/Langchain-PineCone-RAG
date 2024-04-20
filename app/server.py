from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from chain import chain as pinecone_wiki_chain 

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Add the pinecone rag endpoint to point to the chain function 
add_routes(app, pinecone_wiki_chain, path="/pinecone-wiki") 

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
