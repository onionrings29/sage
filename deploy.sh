#!/bin/bash
# Deploy Sage Kinect Service to CT 144 on homelab01
set -e

HOST="root@10.129.20.12"
CT=144
REMOTE_DIR="/opt/sage/kinect-service"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

echo "==> Syncing kinect-service to CT $CT..."
ssh $HOST "pct exec $CT -- mkdir -p $REMOTE_DIR/models"

# Rsync source files (exclude models — downloaded on server)
sshpass -p 'awantasayo' rsync -avz --delete \
    --exclude 'models/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    kinect-service/ root@10.129.20.88:$REMOTE_DIR/

# Download model if missing
echo "==> Checking model..."
ssh $HOST "pct exec $CT -- bash -c '
    if [ ! -f $REMOTE_DIR/models/pose_landmarker.task ]; then
        echo \"Downloading heavy pose model...\"
        wget -q -O $REMOTE_DIR/models/pose_landmarker.task $MODEL_URL
        echo \"Model downloaded (\$(du -h $REMOTE_DIR/models/pose_landmarker.task | cut -f1))\"
    else
        echo \"Model exists (\$(du -h $REMOTE_DIR/models/pose_landmarker.task | cut -f1))\"
    fi
'"

# Restart service
echo "==> Restarting sage-kinect service..."
ssh $HOST "pct exec $CT -- systemctl restart sage-kinect"
sleep 2

# Check status
echo "==> Service status:"
ssh $HOST "pct exec $CT -- systemctl is-active sage-kinect"
echo ""
echo "Dashboard: http://10.129.20.88:8080"
