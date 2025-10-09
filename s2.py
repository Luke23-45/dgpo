import envs.panda_env
import time
import cv2

env = envs.panda_env.PandaEnv()

for i in range(10):
    print(f"--- Episode {i+1} ---")
    obs, _ = env.reset()
    
    # Render and show the first frame to check the camera
    img = obs["image_primary"]
    cv2.imshow("Panda Env Reset", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1000) # Show for 1 second

    # Optional: Run a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

cv2.destroyAllWindows()
env.close()