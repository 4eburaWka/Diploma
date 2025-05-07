git pull

docker build -t glaucoma/be -f prediction/Dockerfile prediction/
docker build -t glaucoma/nginx -f prediction/Dockerfile-nginx prediction/

docker stop glaucoma_be
docker rm glaucoma_be
docker stop glaucoma_nginx
docker rm glaucoma_nginx

docker run -d --network core-network --name glaucoma_be glaucoma/be
docker run -d --network core-network --name glaucoma_nginx glaucoma/nginx