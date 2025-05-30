# ------------------------
# bin/docker_build.sh
# ------------------------
#!/bin/bash
FILE=Gemfile.lock
if [ -f "$FILE" ]; then
    echo "Removing stale Gemfile.lock..."
    rm $FILE
fi

echo "Building Docker image..."
docker build -t "iclr-2025:latest" .


