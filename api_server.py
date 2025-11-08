from flask import Flask, request, jsonify, send_from_directory, abort
import os
import time
from typing import Dict, Any

# Import your system from main.py
from main import VMPlacementSystem, create_sample_data, VirtualMachine

app = Flask(__name__, static_folder='.')

# Initialize system with sample data
nodes, sample_vms = create_sample_data()
system = VMPlacementSystem(nodes)

# Optional: register sample VMs as requests and perform initial optimization
for vm in sample_vms:
    system.add_vm_request(vm)
_ = system.optimize_placement()
_ = system.mine_block()


@app.route("/", methods=["GET"])
def serve_dashboard():
    """Serve the dashboard (dashboard.html) at root."""
    path = os.path.join(os.getcwd(), "dashboard.html")
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(os.getcwd(), "dashboard.html")


@app.route("/favicon.ico")
def favicon():
    # If you have a favicon file in the repo, serve it, otherwise 404
    fpath = os.path.join(os.getcwd(), "favicon.ico")
    if os.path.exists(fpath):
        return send_from_directory(os.getcwd(), "favicon.ico")
    return ("", 204)


# API endpoints --------------------------------------------------------------

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return current system stats (node utilizations, counts)."""
    return jsonify(system.get_system_stats())


@app.route("/api/blockchain", methods=["GET"])
def api_blockchain():
    """Return high-level blockchain summary."""
    chain = system.blockchain.chain
    latest = chain[-1] if chain else None
    return jsonify({
        "blocks": len(chain),
        "latest_block_id": latest.block_id if latest else None,
        "latest_hash": latest.hash if latest else None,
    })


@app.route("/api/blockchain/ledger", methods=["GET"])
def api_blockchain_ledger():
    """Return full ledger: blocks with placement decisions and transactions."""
    ledger = []
    for b in system.blockchain.chain:
        ledger.append({
            "block_id": b.block_id,
            "previous_hash": b.previous_hash,
            "hash": b.hash,
            "nonce": b.nonce,
            "merkle_root": b.merkle_root,
            "timestamp": b.timestamp,
            "transactions": [t.to_dict() for t in b.transactions],
            "placement_decision": b.placement_decision.to_dict() if b.placement_decision else None
        })
    return jsonify(ledger)


@app.route("/api/vm/request", methods=["POST"])
def api_vm_request():
    """Create a new VM request. JSON body must include vm_id, cpu_requirement, memory_requirement,
       storage_requirement, network_requirement, priority (int), owner (str)."""
    data = request.get_json(force=True)
    required = ["vm_id", "cpu_requirement", "memory_requirement", "storage_requirement", "network_requirement", "priority", "owner"]
    if not all(k in data for k in required):
        return jsonify({"error": "missing fields", "required": required}), 400

    vm = VirtualMachine(
        vm_id=str(data["vm_id"]),
        cpu_requirement=float(data["cpu_requirement"]),
        memory_requirement=float(data["memory_requirement"]),
        storage_requirement=float(data["storage_requirement"]),
        network_requirement=float(data["network_requirement"]),
        priority=int(data["priority"]),
        owner=str(data["owner"])
    )
    system.add_vm_request(vm)
    return jsonify({"status": "ok", "vm_id": vm.vm_id}), 201


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """Run optimization and mine a block. Returns placement and mined block info (or a message)."""
    placement = system.optimize_placement()
    block = system.mine_block()

    response: Dict[str, Any] = {"placement": placement}
    if block is None:
        # No block mined (allowed behavior)
        response["block"] = None
        response["message"] = "No block mined (no pending transactions or PoW not achieved)."
        return jsonify(response), 200

    response["block"] = {
        "block_id": block.block_id,
        "hash": block.hash,
        "nonce": block.nonce,
        "merkle_root": block.merkle_root,
        "transactions": [t.to_dict() for t in block.transactions],
        "placement_decision": block.placement_decision.to_dict() if block.placement_decision else None
    }
    return jsonify(response), 201


# Simple health endpoint
@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"ok": True, "time": time.time()})


if __name__ == "__main__":
    # Note: debug mode restarts process twice; set debug=False for single-run behavior
    app.run(host="0.0.0.0", port=5000, debug=True)