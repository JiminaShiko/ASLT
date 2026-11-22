# Text to Sign demo media

Assets used by the Text to Sign panel (`Front-end/user_homepage/translation.html`)
through `Backend/user/text_to_sign.py`. The panel shows a **word sign** when the
word model knows the word, and **fingerspells** the rest from the alphabet here.

```
text_to_sign_demo/
├── alphabet/   A-Z letter signs. A.png ... Z.png, plus J.mp4 and Z.mp4
└── word/       <word>.mp4 for every vocabulary word in Model/videoModel/vocab.json
```

## Adding or replacing a sign

The file name is the word, lower case, with any of these extensions:
`.mp4`, `.webm`, `.gif`, `.webp`, `.png`, `.jpg`, `.jpeg`. Restart is not needed —
the backend re-reads the folders when the vocabulary, manifest or folder changes.

- Word sign: `word/hello.mp4` (shown for "hello", "Hello", "HELLO").
- Letter sign: `alphabet/A.png` (static) or `alphabet/J.mp4` (movement letter).

A word without a file here falls back to the shortest cached ASL Citizen clip that
the browser can decode; if there is none, it is fingerspelled instead.

## Sources and licensing

- **`alphabet/` letter illustrations** — the "Sign language A" … "Sign language Z"
  set on Wikimedia Commons (`https://commons.wikimedia.org/wiki/Category:ASL_letters`),
  released into the **public domain** by wpclipart.com.
  `J.mp4` and `Z.mp4` are rendered from those illustrations: J and Z are the two
  letters that need movement, so they are shipped as clips that show the drawn
  movement arrows rather than a still photo. Replace them with real footage of a
  signer if you have it (same file names).
- **`word/` clips** — one clip per vocabulary word, copied out of the ASL Citizen
  dataset cached locally under `Model/videoModel/ASL_Citizen/videos` (only the
  shortest browser-playable clip of each word was kept).

  ASL Citizen is released under the **Microsoft Research License Terms**
  (non-commercial, non-revenue-generating research use):
  <https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license>.
  Keep that in mind before publishing this folder in a public repository; if the
  course does not need the clips in Git, add `Front-end/text_to_sign_demo/word/`
  to `.gitignore` and copy the clips locally instead.
