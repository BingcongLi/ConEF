In ConEF, count sketch is recomended as the error compressor. Here's a couple of implementation for count sketches.

- CountSketch.py uses a signle hash function. In our experiements, we observe that collisions in hashes are the main cause of compression error. Under the same memory budget, we find that using a single hash function typically works better than three hash functions, since the latter has more collisions in hashing. 

- CountSketchHash3.py uses three hash functions. This is mainly used for comparison, instead of intergrating to real systems.
- halfCountSketch.py uses half precision trick, that is, we store count sketches using fp16 instead of fp32 to save memory more aggressively.
