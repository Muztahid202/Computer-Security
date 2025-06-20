/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  /*
   * Implement this function to read n cache lines.
   * 1. Allocate a buffer of size n * LINESIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
   * 3. Collect total time taken in an array using `performance.now()`.
   * 4. Return the median of the time taken in milliseconds.
   */
  const bufferSize = n * LINESIZE;
  const buffer = new ArrayBuffer(bufferSize);
  const view = new Uint8Array(buffer);
  
  const measurements = [];
  
  for (let iter = 0; iter < 10; iter++) {
    const startTime = performance.now();
    
    for (let i = 0; i < bufferSize; i += LINESIZE) {
      const value = view[i];
      if (value === Infinity) {
        throw new Error("Impossible");
      }
    }
    
    const endTime = performance.now();
    measurements.push(endTime - startTime);
  }
  
  // Improved median calculation
  measurements.sort((a, b) => a - b);
  const mid = Math.floor(measurements.length / 2);
  const median = measurements.length % 2 === 0
    ? (measurements[mid - 1] + measurements[mid]) / 2
    : measurements[mid];
    
  return median;
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
     const results = {};
    const testValues = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000];
    
    for (const n of testValues) {
      try {
        results[n] = readNlines(n);
      } catch (error) {
        console.error(`Failed for n=${n}:`, error);
        break;
      }
    }

    /* Call the readNlines function for n = 1, 10, ... 10,000,000 and store the result */

    self.postMessage(results);
  }
});
