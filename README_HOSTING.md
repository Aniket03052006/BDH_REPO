# Hosting Guide for BDH (Baby Dragon Hatchling) Explorer

To host the **Neural Explorer**, you have several options ranging from free tiers for demoing to robust cloud deployments.

## Option 1: Hugging Face Spaces (Recommended for ML Demos)
Hugging Face Spaces is the best place to host ML-related web apps for free.

1.  Create a new [Hugging Face Space](https://huggingface.co/new-space).
2.  Select **Docker** as the SDK.
3.  Upload all files from the `finalised_model` folder (including the `Dockerfile`).
4.  If you have the large `.pt` weights file, use **Git LFS** to upload it, or ensure `server.py` can load a default if it's missing.
5.  Wait for the build to finish. Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`.

**Note:** Hugging Face automatically handles the exposed port. If it fails to connect, you might need to ensure `server.py` listens on `0.0.0.0`.

---

## Option 2: Railway / Render (General Web Hosting)
These platforms are excellent for simple "Direct from GitHub" deployments.

1.  Connect your GitHub repository to [Railway](https://railway.app/) or [Render](https://render.com/).
2.  The platforms will detect the `Dockerfile` and build it automatically.
3.  Ensure the environment variable `PORT` is set to `8003` if required, or update `server.py` to use `os.environ.get("PORT", 8003)`.
4.  **Important:** Free tiers usually have limited RAM. PyTorch is heavy; if the app crashes with "Out of Memory", you may need a paid tier or to use `torch.set_num_threads(1)` to reduce memory footprint.

---

## Option 3: Local Tunnel (For Quick Sharing)
If you just want to show the app to someone without "real" hosting:

1.  Run the server locally: `python server.py`
2.  In a new terminal, use **ngrok** (or `cloudflared`):
    ```bash
    ngrok http 8003
    ```
3.  Share the provided URL. Note that the frontend `main.js` might need to be aware of the tunnel URL for WebSocket connections.

---

## Configuration Edits for Hosting
For best results in a hosted environment, consider these minor code changes:

### 1. Dynamic Port in `server.py`
Update the bottom of `server.py` to use an environment variable:
```python
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 2. WebSocket URL in `main.js`
In `main.js`, ensure the `WS_URL` correctly points to the hosting domain (usually `wss://` for production).

---

## Option 4: Split Deployment (Vercel Frontend + Render Backend)
This is a standard setup, separating static assets from the compute-heavy model.

### 1. Backend (Render)
1. Deploy the `BDH_Explorer` repo to Render as a **Web Service** (using Docker).
2. Note your Render URL (e.g., `https://bdh-backend.onrender.com`).

### 2. Frontend (Vercel)
1. Open `main.js` and find the `BACKEND_HOST` variable.
2. Change it from `null` to your Render domain (without `https://`):
   ```javascript
   const BACKEND_HOST = 'bdh-backend.onrender.com';
   ```
3. Commit and push this change.
4. On [Vercel](https://vercel.com/), click **Add New > Project**.
5. Connect your GitHub repo.
6. Vercel will host it as a static site. Your dashboard will be live at `https://your-app.vercel.app`.

**Note**: `server.py` already includes CORS support to allow the Vercel domain.
