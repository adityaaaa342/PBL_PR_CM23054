/**
 * Frontend Controller for Community Detection Web Application
 * Handles Vis.js Graph Rendering, Physics, API Requests, and Modals
 */

let network = null;
let networkData = { nodes: new vis.DataSet([]), edges: new vis.DataSet([]) };
let physicsEnabled = true;

document.addEventListener("DOMContentLoaded", () => {
    initApp();
    setupEventListeners();
    runDetection(); // Initial run
});

function initApp() {
    const container = document.getElementById("network-canvas");
    
    const options = {
        nodes: {
            shape: "dot",
            borderWidth: 1.5,
            shadow: { enabled: true, color: "rgba(0,0,0,0.6)", size: 10, x: 2, y: 2 },
            font: { color: "#FFFFFF", size: 10, face: "Outfit" }
        },
        edges: {
            smooth: { type: "continuous", roundness: 0.2 },
            selectionWidth: 2.5
        },
        physics: {
            enabled: true,
            solver: "forceAtlas2Based",
            forceAtlas2Based: {
                gravitationalConstant: -40,
                centralGravity: 0.008,
                springLength: 70,
                springConstant: 0.12,
                damping: 0.6
            },
            stabilization: { iterations: 150, updateInterval: 25 }
        },
        interaction: {
            hover: true,
            tooltipDelay: 100,
            navigationButtons: false,
            keyboard: true
        }
    };

    network = new vis.Network(container, networkData, options);
}

function setupEventListeners() {
    // Dataset Change Listener
    const datasetSelect = document.getElementById("dataset-select");
    datasetSelect.addEventListener("change", () => {
        updateDatasetMeta(datasetSelect.value);
        runDetection();
    });

    // Algorithm Radio Buttons
    const algoOptions = document.querySelectorAll(".algo-option");
    algoOptions.forEach(opt => {
        opt.addEventListener("click", () => {
            algoOptions.forEach(o => o.classList.remove("active"));
            opt.classList.add("active");
            const radio = opt.querySelector("input");
            radio.checked = true;
            toggleHyperparams(radio.value);
        });
    });

    // Resolution slider
    const resSlider = document.getElementById("param-resolution");
    if (resSlider) {
        resSlider.addEventListener("input", (e) => {
            document.getElementById("res-val").textContent = e.target.value;
        });
    }

    // Run Detection Button
    document.getElementById("btn-run").addEventListener("click", () => {
        runDetection();
    });

    // Physics Freeze Toggle
    const btnFreeze = document.getElementById("btn-freeze-physics");
    btnFreeze.addEventListener("click", () => {
        physicsEnabled = !physicsEnabled;
        network.setOptions({ physics: { enabled: physicsEnabled } });
        btnFreeze.innerHTML = physicsEnabled 
            ? '<i class="fa-solid fa-pause"></i> Freeze Physics' 
            : '<i class="fa-solid fa-play"></i> Resume Physics';
    });

    // Fit View
    document.getElementById("btn-fit-zoom").addEventListener("click", () => {
        network.fit({ animation: { duration: 600, easingFunction: "easeInOutQuad" } });
    });

    // Benchmark Modal
    const modalBenchmark = document.getElementById("modal-benchmark");
    document.getElementById("btn-benchmark").addEventListener("click", () => {
        modalBenchmark.style.display = "flex";
        runBenchmarkComparison();
    });
    document.getElementById("close-benchmark-modal").addEventListener("click", () => {
        modalBenchmark.style.display = "none";
    });

    // Viva Modal
    const modalViva = document.getElementById("modal-viva");
    document.getElementById("btn-viva-modal").addEventListener("click", () => {
        modalViva.style.display = "flex";
    });
    document.getElementById("close-viva-modal").addEventListener("click", () => {
        modalViva.style.display = "none";
    });

    // Upload Modal
    const modalUpload = document.getElementById("modal-upload");
    document.getElementById("btn-upload-modal").addEventListener("click", () => {
        modalUpload.style.display = "flex";
    });
    document.getElementById("close-upload-modal").addEventListener("click", () => {
        modalUpload.style.display = "none";
    });

    // Screenshot button
    document.getElementById("btn-export-png").addEventListener("click", takeScreenshot);

    // CSV File Upload
    const fileInput = document.getElementById("csv-file-input");
    fileInput.addEventListener("change", handleFileUpload);

    // Close modals when clicking backdrop
    window.addEventListener("click", (e) => {
        if (e.target.classList.contains("modal-backdrop")) {
            e.target.style.display = "none";
        }
    });
}

function toggleHyperparams(algo) {
    const louvainBox = document.getElementById("louvain-params");
    const targetKBox = document.getElementById("target-k-params");
    
    louvainBox.style.display = (algo === "louvain") ? "flex" : "none";
    targetKBox.style.display = (algo === "girvan_newman" || algo === "spectral") ? "flex" : "none";
}

function updateDatasetMeta(datasetId) {
    const metaMap = {
        "football": { nodes: 115, edges: 613, gt: "12 Conferences" },
        "facebook": { nodes: 78, edges: 413, gt: "4 Circles" },
        "coauthorship": { nodes: 120, edges: 995, gt: "3 Domains" }
    };
    const meta = metaMap[datasetId] || { nodes: "?", edges: "?", gt: "Unsupervised" };
    document.getElementById("meta-nodes").textContent = meta.nodes;
    document.getElementById("meta-edges").textContent = meta.edges;
    document.getElementById("meta-gt").textContent = meta.gt;
}

async function runDetection() {
    const dataset = document.getElementById("dataset-select").value;
    const algoInput = document.querySelector('input[name="algo"]:checked');
    const algorithm = algoInput ? algoInput.value : "louvain";
    const resolution = document.getElementById("param-resolution").value;
    const targetK = document.getElementById("param-target-k").value;
    
    const loading = document.getElementById("canvas-loading");
    if (loading) loading.style.display = "flex";

    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset: dataset,
                algorithm: algorithm,
                resolution: resolution,
                target_k: targetK
            })
        });

        const data = await response.json();
        if (!data.success) {
            alert("Error running detection: " + (data.error || "Unknown"));
            return;
        }

        // Update Metrics Cards
        animateCounter("val-comms", data.num_communities);
        document.getElementById("val-modularity").textContent = data.metrics.modularity.toFixed(4);
        document.getElementById("val-nmi").textContent = (data.metrics.nmi !== null) ? `${(data.metrics.nmi * 100).toFixed(1)}%` : "N/A";
        document.getElementById("val-conductance").textContent = data.metrics.conductance.toFixed(4);
        document.getElementById("val-runtime").textContent = `${data.metrics.time_ms.toFixed(1)} ms`;

        // Update Community Breakdown Sidebar
        renderCommunityBreakdown(data.community_breakdown);

        // Update Graph Canvas
        networkData.nodes.clear();
        networkData.edges.clear();
        networkData.nodes.add(data.graph_data.nodes);
        networkData.edges.add(data.graph_data.edges);

        setTimeout(() => {
            network.fit();
            if (loading) loading.style.display = "none";
        }, 300);

    } catch (err) {
        console.error("API error:", err);
        if (loading) loading.style.display = "none";
    }
}

function renderCommunityBreakdown(breakdown) {
    const container = document.getElementById("communities-list");
    container.innerHTML = "";
    
    breakdown.forEach(item => {
        const pill = document.createElement("div");
        pill.className = "comm-pill-item";
        pill.innerHTML = `
            <span><span class="comm-color-dot" style="background-color: ${item.color};"></span> Community #${item.id}</span>
            <span class="meta-v">${item.size} nodes</span>
        `;
        container.appendChild(pill);
    });
}

function animateCounter(id, targetVal) {
    const el = document.getElementById(id);
    let curr = 0;
    const step = Math.max(1, Math.floor(targetVal / 10));
    const interval = setInterval(() => {
        curr += step;
        if (curr >= targetVal) {
            el.textContent = targetVal;
            clearInterval(interval);
        } else {
            el.textContent = curr;
        }
    }, 25);
}

async function runBenchmarkComparison() {
    const dataset = document.getElementById("dataset-select").value;
    const tbody = document.getElementById("benchmark-tbody");
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px;">Running multi-algorithm benchmark...</td></tr>`;

    try {
        const res = await fetch("/api/benchmark", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset: dataset })
        });
        const data = await res.json();
        
        tbody.innerHTML = "";
        data.benchmark.forEach(row => {
            const tr = document.createElement("tr");
            const isBestMod = (row.modularity >= 0.6);
            tr.innerHTML = `
                <td><b>${row.algorithm}</b></td>
                <td>${getMethodology(row.algorithm)}</td>
                <td><span class="badge-usn">${row.communities}</span></td>
                <td style="color:${isBestMod ? '#00FF66' : '#F3F4F6'}">${row.modularity.toFixed(4)}</td>
                <td>${row.conductance.toFixed(4)}</td>
                <td style="color:#00F0FF">${row.nmi ? (row.nmi * 100).toFixed(1) + '%' : 'N/A'}</td>
                <td style="color:#F97316">${row.time_ms.toFixed(2)} ms</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" style="color:red; text-align:center;">Failed to run benchmark.</td></tr>`;
    }
}

function getMethodology(algo) {
    if (algo.includes("Louvain")) return "Modularity Maximization";
    if (algo.includes("Girvan")) return "Edge Betweenness Divisive";
    if (algo.includes("Label")) return "Linear Dynamic Consensus";
    return "Laplacian Eigenvectors";
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append("file", file);
    
    const statusBox = document.getElementById("upload-status");
    statusBox.textContent = "Uploading and building graph topology...";
    
    try {
        const res = await fetch("/api/upload_csv", { method: "POST", body: formData });
        const data = await res.json();
        if (data.success) {
            statusBox.innerHTML = `<span style="color:#00FF66;">Uploaded successfully! ${data.nodes} nodes, ${data.edges} edges.</span>`;
            
            // Add custom option to dropdown
            const select = document.getElementById("dataset-select");
            const opt = document.createElement("option");
            opt.value = data.dataset_id;
            opt.textContent = `📁 ${data.name} (${data.nodes} nodes)`;
            select.appendChild(opt);
            select.value = data.dataset_id;
            
            setTimeout(() => {
                document.getElementById("modal-upload").style.display = "none";
                runDetection();
            }, 800);
        } else {
            statusBox.innerHTML = `<span style="color:#FF453A;">Error: ${data.error}</span>`;
        }
    } catch (err) {
        statusBox.innerHTML = `<span style="color:#FF453A;">Upload failed.</span>`;
    }
}

function takeScreenshot() {
    const canvas = document.querySelector("#network-canvas canvas");
    if (!canvas) return;
    const imgURI = canvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.download = `community_graph_${Date.now()}.png`;
    link.href = imgURI;
    link.click();
}
