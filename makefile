.PHONY: dockerbuild
dockerbuild:
	bash -c "sudo docker build -t app ."

.PHONY: dockerrun
dockerrun:
	bash -c "sudo docker compose up"

.PHONY: docker
docker:
	$(MAKE) dockerbuild
	$(MAKE) dockerrun