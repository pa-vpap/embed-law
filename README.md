#build docker image from root folder

docker build --no-cache -t roberta-v3 -f Dockerfile2 .

#run docker instance

docker run -it -p 8080:8080 roberta-v3 


#check from another terminal with
 
curl -X POST http://localhost:8080/v1/embeddings -H "Content-Type: application/json" -d '{
    "input": "Αυτό είναι ένα δοκιμαστικό κείμενο.",
    "model": "AI-team-UoA/GreekLegalRoBERTa_v3"


#use it from another application

EMBEDDING_BINDING=openai
EMBEDDING_MODEL=AI-team-UoA/GreekLegalRoBERTa_v3
EMBEDDING_DIM=768
EMBEDDING_BINDING_HOST=http://127.0.0.1:8080/v1

#use it from another docker instance

EMBEDDING_BINDING=openai
EMBEDDING_MODEL=AI-team-UoA/GreekLegalRoBERTa_v3
EMBEDDING_DIM=768
EMBEDDING_BINDING_HOST=http://host.docker.internal:8080/v1
