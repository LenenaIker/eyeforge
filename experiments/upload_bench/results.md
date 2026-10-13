Upload bench, 69 photos from grass (phone originals, 4032x2268), 10 Mbit/s uplink, local server (2026-10-13)

| case | MB | batches | seconds | server read | save | library | server total |
|---|---|---|---|---|---|---|---|
| orig-40x1 | 337 | 9 | 270.1 | 269.03 | 0.17 | 0.89 | 270.09 |
| orig-8x3 | 337 | 68 | 280.3 | 802.84 | 0.24 | 1.99 | 805.12 |
| small-40x1 | 52 | 2 | 42.5 | 41.93 | 0.01 | 0.57 | 42.52 |
| small-8x3 | 52 | 7 | 43.5 | 113.16 | 0.01 | 0.54 | 113.72 |

orig-8x3 ran before the fix that gives the first batch the whole uplink (about 13 s too slow). "server read" is
the time from the request's arrival to the handler, so it holds the upload itself; with batches side by side
the server times add up over the batches. What the server does with the photos (save + library) is under 1 s.
