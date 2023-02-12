docker run --privileged \
    --runtime nvidia --rm \
    -p 5000:5000 -p 443:443 -p 80:80 \
    -v /home/usernano/face_id/known_faces:/known_faces \
    -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra \
    --name face_id \
    --device=/dev/nvhost-ctrl \
    --device=/dev/nvhost-ctrl-gpu \
    --device=/dev/nvhost-prof-gpu \
    --device=/dev/nvmap \
    --device=/dev/nvhost-gpu \
    --device=/dev/nvhost-as-gpu \
    ivanhahan/face_id:jetson