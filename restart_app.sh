git pull

docker build -t glaucoma/be -f prediction/Dockerfile prediction/
docker build -t glaucoma/nginx -f prediction/Dockerfile-nginx prediction/

docker stop glaucoma_be || true
docker rm glaucoma_be || true
docker stop glaucoma_nginx || true
docker rm glaucoma_nginx || true

docker run -d --network core-network --name glaucoma_be glaucoma/be
docker run -d --network core-network --name glaucoma_nginx glaucoma/nginx