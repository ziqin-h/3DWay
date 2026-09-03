(() => {
  const copyBtn = document.getElementById("copy-bibtex");
  const bibtex = document.getElementById("bibtex");
  if (copyBtn && bibtex) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(bibtex.textContent);
        copyBtn.textContent = "Copied";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 1600);
      } catch (e) {
        copyBtn.textContent = "Select & copy";
      }
    });
  }

  const videos = Array.from(document.querySelectorAll(".pipeline-video"));
  if (!videos.length) return;

  const HOLD_MS = 3000;
  let active = 0;
  let running = false;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const pauseAllAtStart = () => {
    videos.forEach((video) => {
      video.pause();
      try {
        video.currentTime = 0;
      } catch (e) {}
    });
  };

  const playOne = (video) =>
    new Promise((resolve) => {
      const finish = () => {
        video.removeEventListener("ended", finish);
        try {
          video.pause();
          if (video.duration && Number.isFinite(video.duration)) {
            video.currentTime = Math.max(0, video.duration - 0.05);
          }
        } catch (e) {}
        resolve();
      };

      video.addEventListener("ended", finish, { once: true });
      const playPromise = video.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => finish());
      }
    });

  const runSequence = async () => {
    if (running) return;
    running = true;
    pauseAllAtStart();
    for (let i = 0; i < videos.length; i += 1) {
      active = i;
      await playOne(videos[i]);
    }
    await sleep(HOLD_MS);
    running = false;
    runSequence();
  };

  Promise.all(
    videos.map(
      (video) =>
        new Promise((resolve) => {
          if (video.readyState >= 1) {
            resolve();
            return;
          }
          video.addEventListener("loadedmetadata", resolve, { once: true });
          video.addEventListener("error", resolve, { once: true });
        })
    )
  ).then(runSequence);
})();
