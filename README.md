# Deployment

The app can be pulled with `docker pull ivanhahan/face_id:jetson`.
Or it can be built manually using `docker build -t ivanhahan/face_id:jetson -f Dockerfile.jetson .`.

The docker image can be run then using:

      `docker run --privileged \
        --runtime nvidia --rm \
        -p 5000:5000 -p 443:443 -p 80:80 \
        -v <known_faces_dir>:/known_faces \
        -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra \
        --name face_id \
        --device=/dev/nvhost-ctrl \
        --device=/dev/nvhost-ctrl-gpu \
        --device=/dev/nvhost-prof-gpu \
        --device=/dev/nvmap \
        --device=/dev/nvhost-gpu \
        --device=/dev/nvhost-as-gpu \
        ivanhahan/face_id:jetson
    `

`known_faces_dir` volume should have the following structure:

    `
    known_faces/
    └── dima
        ├── 5fb16df54b99a.png
        ├── 5fb16df7e1a92.png
        ├── 5fb16df9c3e0c.png
    └── ivan
        ├── 5fb16df54b99a.jpg
        ├── 5fb16df7e1a92.jpg
        ├── 5fb16df9c3e0c.png
        ├── 5fb16dfb0489f.png
    `

Each folder is associated with the specific person and should contain photos of him/her. The number of photos
is not restricted and can be expanded.

# API

API docs can be accessed on `/apidocs` endpoint of the deployed app.