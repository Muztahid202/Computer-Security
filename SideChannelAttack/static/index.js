function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    traceStats: [], // Store statistics for each heatmap
    // Local collection of heapmap images
    heatmaps: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,
    predictedWebsite: "",


    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        //printing the results to console
        console.log("Latency Results:", this.latencyResults);

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
      /* 
       * Implement this function to collect trace data.
       * 1. Create a worker to run the sweep function.
       * 2. Collect the trace data from the worker.
       * 3. Send the trace data to the backend for temporary storage and heatmap generation.
       * 4. Fetch the heatmap from the backend and add it to the local collection.
       * 5. Handle errors and update the status.
       */
      this.isCollecting = true;
      this.status = "Collecting trace data...";
      this.statusIsError = false;
      this.showingTraces = true;

      try {
        let worker = new Worker("worker.js");

        const trace = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        if (trace.error) {
          throw new Error(trace.error);
        }

        this.traceData.push(trace);

        const response = await fetch('/collect_trace', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ trace }),
        });

        const data = await response.json();
        if (response.status !== 200) {
          throw new Error(data.error || 'Failed to collect trace');
        }

        // Save predicted website
        this.predictedWebsite = data.prediction || "Unknown";


        //print the response to console
        console.log("Trace Collection Response:", data);

        // Store the heatmap path
        this.heatmaps.push(data.heatmap);

        // Store the stats (optional UI rendering later)
        this.traceStats.push(data.stats);

        this.status = "Trace collection complete!";

        worker.terminate();
      } catch (error) {
        console.error("Error collecting trace:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
      /* 
       * Implement this function to download the trace data.
       * 1. Fetch the latest data from the backend API.
       * 2. Create a download file with the trace data in JSON format.
       * 3. Handle errors and update the status.
       */
      try {
        this.status = "Fetching trace data from server...";

        const response = await fetch('/api/get_results');
        if (!response.ok) {
          throw new Error('Failed to fetch traces from server.');
        }

        const data = await response.json();

        const blob = new Blob([JSON.stringify(data, null, 2)], {
          type: 'application/json',
        });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `traces_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.status = "Trace download complete!";
      } catch (error) {
        console.error("Error downloading traces:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Clear all results from the server
    async clearResults() {
      /* 
       * Implement this function to clear all results from the server.
       * 1. Send a request to the backend API to clear all results.
       * 2. Clear local copies of trace data and heatmaps.
       * 3. Handle errors and update the status.
       */
      try {
        this.status = "Clearing results...";
        const response = await fetch('/api/clear_results', {
          method: 'POST',
        });

        if (!response.ok) {
          throw new Error('Failed to clear results');
        }

        // Clear local state
        this.heatmaps = [];
        this.traceData = [];
        this.traceStats = [];
        //added later
        this.latencyResults = null;
        this.showingTraces = false;
        this.isCollecting = false;
        //added later
        this.status = "Cleared!";
      } catch (error) {
        console.error("Error clearing results:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      }
    },
  };
}
