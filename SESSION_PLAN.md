# Kinect camera-only recovery — 2026-10-25

Owner requested disabling Kinect pose inference while retaining the camera feed. This is separate from Tapo object detection, which must remain active on Frigate's GPU.

- [x] Confirm local source matches deployed CT144 source; use isolated worktree.
- [x] Regression tests observed red against original capture loop and status.
- [x] Implement camera-only default; skip MediaPipe import/model/inference; expose disabled pose explicitly without reporting false absence.
- [x] Verify local tests and compilation.
- [ ] Independent review; address findings.
- [ ] Merge reviewed source, deploy complete slice, verify camera bytes/advancing frames, disabled pose and CPU reduction.
- [ ] Update service documentation and incident record.

During user's Jellyfin concurrency test: keep Frigate/Jellyfin and GPU driver unchanged. Camera-only preparation is local until reviewed.
