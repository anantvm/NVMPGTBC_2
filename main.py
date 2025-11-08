import numpy as np
import hashlib
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from datetime import datetime
import uuid
import copy
import os


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

@dataclass
class VirtualMachine:
    vm_id: str
    cpu_requirement: float
    memory_requirement: float
    storage_requirement: float
    network_requirement: float
    priority: int
    owner: str

    def get_resource_vector(self) -> np.ndarray:
        return np.array([
            self.cpu_requirement,
            self.memory_requirement,
            self.storage_requirement,
            self.network_requirement
        ])

@dataclass
class PhysicalNode:
    node_id: str
    cpu_capacity: float
    memory_capacity: float
    storage_capacity: float
    network_capacity: float
    energy_cost: float
    location: str

    cpu_used: float = 0.0
    memory_used: float = 0.0
    storage_used: float = 0.0
    network_used: float = 0.0

    def get_capacity_vector(self) -> np.ndarray:
        return np.array([
            self.cpu_capacity,
            self.memory_capacity,
            self.storage_capacity,
            self.network_capacity
        ])

    def get_usage_vector(self) -> np.ndarray:
        return np.array([
            self.cpu_used,
            self.memory_used,
            self.storage_used,
            self.network_used
        ])

    def get_available_vector(self) -> np.ndarray:
        return self.get_capacity_vector() - self.get_usage_vector()

    def can_host(self, vm: VirtualMachine) -> bool:
        available = self.get_available_vector()
        required = vm.get_resource_vector()
        return np.all(available >= required)

    def allocate_vm(self, vm: VirtualMachine):
        if self.can_host(vm):
            self.cpu_used += vm.cpu_requirement
            self.memory_used += vm.memory_requirement
            self.storage_used += vm.storage_requirement
            self.network_used += vm.network_requirement
            return True
        return False

    def deallocate_vm(self, vm: VirtualMachine):
        self.cpu_used = max(0, self.cpu_used - vm.cpu_requirement)
        self.memory_used = max(0, self.memory_used - vm.memory_requirement)
        self.storage_used = max(0, self.storage_used - vm.storage_requirement)
        self.network_used = max(0, self.network_used - vm.network_requirement)

@dataclass
class Transaction:
    tx_id: str
    vm_id: str
    node_id: str
    action: str
    timestamp: float
    requester: str
    gas_fee: float
    efficiency_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'tx_id': self.tx_id,
            'vm_id': self.vm_id,
            'node_id': self.node_id,
            'action': self.action,
            'timestamp': self.timestamp,
            'requester': self.requester,
            'gas_fee': self.gas_fee,
            'efficiency_score': self.efficiency_score
        }

@dataclass
class PlacementDecision:
    decision_id: str
    placement: Dict[str, str]
    timestamp: float
    overall_efficiency: float
    resource_utilization: float
    energy_efficiency: float
    load_balance_score: float

    def to_dict(self) -> Dict:
        return {
            'decision_id': self.decision_id,
            'placement': self.placement,
            'timestamp': self.timestamp,
            'overall_efficiency': self.overall_efficiency,
            'resource_utilization': self.resource_utilization,
            'energy_efficiency': self.energy_efficiency,
            'load_balance_score': self.load_balance_score
        }

@dataclass
class Block:
    block_id: str
    previous_hash: str
    timestamp: float
    transactions: List[Transaction]
    merkle_root: str
    placement_decision: Optional[PlacementDecision] = None
    nonce: int = 0
    hash: str = ""

    def calculate_merkle_root(self) -> str:
        if not self.transactions:
            return hashlib.sha256(b"").hexdigest()

        tx_hashes = [hashlib.sha256(json.dumps(tx.to_dict(), sort_keys=True).encode()).hexdigest()
                     for tx in self.transactions]

        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])

            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i+1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = new_hashes

        return tx_hashes[0]

    def calculate_hash(self) -> str:
        block_data = {
            'block_id': self.block_id,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce,
            'placement_decision': self.placement_decision.to_dict() if self.placement_decision else None
        }
        return hashlib.sha256(json.dumps(block_data, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "block_id": self.block_id,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "transactions": [t.to_dict() for t in self.transactions],
            "merkle_root": self.merkle_root,
            "placement_decision": self.placement_decision.to_dict() if self.placement_decision else None,
            "nonce": self.nonce,
            "hash": self.hash
        }

    @staticmethod
    def from_dict(d: Dict) -> 'Block':
        txs = [Transaction(**t) for t in d.get("transactions", [])]
        pd = d.get("placement_decision")
        placement_decision = PlacementDecision(**pd) if pd else None
        blk = Block(
            block_id=d["block_id"],
            previous_hash=d["previous_hash"],
            timestamp=d["timestamp"],
            transactions=txs,
            merkle_root=d.get("merkle_root", ""),
            placement_decision=placement_decision,
            nonce=d.get("nonce", 0),
            hash=d.get("hash", "")
        )
        return blk

class Blockchain:
    def __init__(self, difficulty: int = 4, persist_file: Optional[str] = None):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.persist_file = persist_file
        if self.persist_file and os.path.exists(self.persist_file):
            try:
                self._load_from_file(self.persist_file)
            except Exception:
                self.create_genesis_block()
        else:
            self.create_genesis_block()

    def create_genesis_block(self):
        genesis = Block(
            block_id="0",
            previous_hash="0",
            timestamp=time.time(),
            transactions=[],
            merkle_root="",
            nonce=0
        )
        genesis.merkle_root = genesis.calculate_merkle_root()
        genesis.hash = genesis.calculate_hash()
        self.chain = [genesis]
        self.pending_transactions = []
        if self.persist_file:
            self._save_to_file(self.persist_file)

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_transaction(self, transaction: Transaction):
        existing_ids = {t.tx_id for b in self.chain for t in b.transactions}
        existing_ids.update(t.tx_id for t in self.pending_transactions)
        if transaction.tx_id in existing_ids:
            transaction.tx_id = f"{transaction.tx_id}_{int(time.time() * 1000)}"
        self.pending_transactions.append(transaction)
        if self.persist_file:
            self._save_to_file(self.persist_file)

    def mine_pending_transactions(self, placement_decision: Optional[PlacementDecision] = None, max_nonce: int = 10_000_000, progress_interval: int = 100000) -> Optional[Block]:
        if not self.pending_transactions and placement_decision is None:
            return None

        txs = self.pending_transactions.copy()

        block = Block(
            block_id=str(len(self.chain)),
            previous_hash=self.get_latest_block().hash,
            timestamp=time.time(),
            transactions=txs,
            merkle_root="",
            placement_decision=placement_decision
        )

        block.merkle_root = block.calculate_merkle_root()
        target = "0" * self.difficulty

        for _ in range(max_nonce):
            current_hash = block.calculate_hash()
            if current_hash.startswith(target):
                block.hash = current_hash
                break
            block.nonce += 1
            if block.nonce % progress_interval == 0:
                pass

        if block.hash == "":
            return None

        self.chain.append(block)
        self.pending_transactions = []
        if self.persist_file:
            self._save_to_file(self.persist_file)
        return block

    def is_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def get_best_placement_from_ledger(self) -> Optional[PlacementDecision]:
        best_decision = None
        best_efficiency = -np.inf
        for block in self.chain:
            if block.placement_decision and block.placement_decision.overall_efficiency > best_efficiency:
                best_efficiency = block.placement_decision.overall_efficiency
                best_decision = block.placement_decision
        return best_decision

    def _save_to_file(self, path: str):
        data = {
            "chain": [b.to_dict() for b in self.chain],
            "pending_transactions": [t.to_dict() for t in self.pending_transactions]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def _load_from_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chain = [Block.from_dict(b) for b in data.get("chain", [])]
        self.pending_transactions = [Transaction(**t) for t in data.get("pending_transactions", [])]

class GameTheoryOptimizer:
    def __init__(self, nodes: List[PhysicalNode]):
        self.nodes = nodes

    def _clone_nodes(self) -> List[PhysicalNode]:
        return [copy.deepcopy(n) for n in self.nodes]

    def calculate_utility(self, vm: VirtualMachine, candidate_node: PhysicalNode, all_nodes: List[PhysicalNode]) -> float:
        capacity = candidate_node.get_capacity_vector()
        usage_after = candidate_node.get_usage_vector() + vm.get_resource_vector()
        util = np.divide(usage_after, capacity, out=np.zeros_like(usage_after), where=capacity != 0)
        optimal_util = 0.8
        penalty = np.mean(np.abs(util - optimal_util))
        resource_score = max(0.0, 1.0 - penalty)

        resource_sum = float(np.sum(vm.get_resource_vector()))
        if resource_sum <= 0:
            energy_score = 0.0
        else:
            raw_values = []
            for n in all_nodes:
                safe_cost = n.energy_cost if n.energy_cost > 0 else 1e-9
                raw_values.append(1.0 / (safe_cost * resource_sum))
            raw_min, raw_max = min(raw_values), max(raw_values)
            cand_raw = raw_values[all_nodes.index(candidate_node)]
            if raw_max > raw_min:
                energy_score = (cand_raw - raw_min) / (raw_max - raw_min)
            else:
                energy_score = 1.0

        ratios = []
        for n in all_nodes:
            c = n.get_capacity_vector()
            u = n.get_usage_vector()
            r = np.divide(u, c, out=np.zeros_like(u), where=c != 0)
            ratios.append(float(np.mean(r)))
        variance = float(np.var(ratios))
        load_score = 1.0 / (1.0 + variance)

        weights = [0.45, 0.25, 0.30]
        utility = weights[0]*resource_score + weights[1]*energy_score + weights[2]*load_score
        return float(max(0.0, min(1.0, utility)))

    def nash_equilibrium_placement(self, vms: List[VirtualMachine]) -> Dict[str, str]:
        placement: Dict[str, str] = {}
        remaining_vms = sorted(vms, key=lambda x: x.priority, reverse=True)
        sim_nodes = self._clone_nodes()
        for vm in remaining_vms:
            best_node = None
            best_utility = -1.0
            for idx, node in enumerate(sim_nodes):
                if node.can_host(vm):
                    candidate_nodes = [copy.deepcopy(n) for n in sim_nodes]
                    candidate_nodes[idx].allocate_vm(vm)
                    util = self.calculate_utility(vm, candidate_nodes[idx], candidate_nodes)
                    if util > best_utility:
                        best_utility = util
                        best_node = idx
            if best_node is not None and best_utility > 0:
                sim_nodes[best_node].allocate_vm(vm)
                placement[vm.vm_id] = sim_nodes[best_node].node_id
        for vm_id, node_id in placement.items():
            vm = next((v for v in vms if v.vm_id == vm_id), None)
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            if vm and node:
                node.allocate_vm(vm)
        return placement

    def pareto_optimal_solutions(self, vms: List[VirtualMachine]) -> List[Dict[str, str]]:
        solutions = []
        strategies = [self._greedy_resource_placement, self._greedy_energy_placement, self._balanced_placement]
        for strategy in strategies:
            cloned = self._clone_nodes()
            placement = strategy(vms, cloned)
            if placement:
                solutions.append(placement)
        return solutions

    def _greedy_resource_placement(self, vms: List[VirtualMachine], nodes: List[PhysicalNode]) -> Dict[str, str]:
        placement = {}
        for vm in vms:
            candidates = [n for n in nodes if n.can_host(vm)]
            best = None
            best_score = -1.0
            for n in candidates:
                cap = n.get_capacity_vector()
                usage_after = n.get_usage_vector() + vm.get_resource_vector()
                util = np.divide(usage_after, cap, out=np.zeros_like(usage_after), where=cap != 0)
                node_score = 1.0 - np.mean(np.abs(util - 0.8))
                if node_score > best_score:
                    best_score = node_score
                    best = n
            if best:
                best.allocate_vm(vm)
                placement[vm.vm_id] = best.node_id
        return placement

    def _greedy_energy_placement(self, vms: List[VirtualMachine], nodes: List[PhysicalNode]) -> Dict[str, str]:
        placement = {}
        for vm in vms:
            candidates = [n for n in nodes if n.can_host(vm)]
            if not candidates:
                continue
            raw = []
            rsrc = max(1e-9, float(np.sum(vm.get_resource_vector())))
            for n in candidates:
                safe = n.energy_cost if n.energy_cost > 0 else 1e-9
                raw.append(1.0 / (safe * rsrc))
            rmin, rmax = min(raw), max(raw)
            best = None
            best_score = -1.0
            for idx, n in enumerate(candidates):
                score = (raw[idx]-rmin)/(rmax-rmin) if rmax>rmin else 1.0
                if score > best_score:
                    best_score = score
                    best = n
            if best:
                best.allocate_vm(vm)
                placement[vm.vm_id] = best.node_id
        return placement

    def _balanced_placement(self, vms: List[VirtualMachine], nodes: List[PhysicalNode]) -> Dict[str, str]:
        placement = {}
        for vm in vms:
            candidates = [n for n in nodes if n.can_host(vm)]
            best = None
            best_score = -1.0
            for n in candidates:
                temp_nodes = [copy.deepcopy(x) for x in nodes]
                tn = next(x for x in temp_nodes if x.node_id == n.node_id)
                tn.allocate_vm(vm)
                ratios = []
                for m in temp_nodes:
                    c = m.get_capacity_vector()
                    u = m.get_usage_vector()
                    r = np.divide(u, c, out=np.zeros_like(u), where=c != 0)
                    ratios.append(float(np.mean(r)))
                score = 1.0/(1.0+float(np.var(ratios)))
                if score > best_score:
                    best_score = score
                    best = n
            if best:
                best.allocate_vm(vm)
                placement[vm.vm_id] = best.node_id
        return placement

class VMPlacementSystem:
    def __init__(self, nodes: List[PhysicalNode]):
        self.nodes = nodes
        self.vms = {}
        # Persist blockchain to disk so ledger survives restarts
        self.blockchain = Blockchain(difficulty=2, persist_file="blockchain.json")
        self.optimizer = GameTheoryOptimizer(nodes)
        self.placement_map = {}

    def calculate_placement_efficiency(self, placement: Dict[str, str]) -> Tuple[float, float, float, float]:
        if not placement:
            return 0.0, 0.0, 0.0, 0.0

        total_utilization = 0.0
        utilization_scores = []
        for node in self.nodes:
            capacity = node.get_capacity_vector()
            usage = node.get_usage_vector()
            if np.any(capacity > 0):
                utilization = np.divide(usage, capacity, out=np.zeros_like(usage), where=capacity != 0)
                node_util = np.mean(utilization)
                utilization_scores.append(node_util)
                total_utilization += node_util

        avg_resource_utilization = total_utilization / len(self.nodes) if self.nodes else 0.0

        total_energy_cost = 0.0
        for vm_id, node_id in placement.items():
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            if node:
                total_energy_cost += node.energy_cost

        avg_energy_cost = total_energy_cost / len(placement) if placement else 0.0
        energy_efficiency = 1.0 / (1.0 + avg_energy_cost) if avg_energy_cost >= 0 else 0.0

        load_balance_variance = np.var(utilization_scores) if len(utilization_scores) > 1 else 0.0
        load_balance_score = 1.0 / (1.0 + load_balance_variance)

        weights = [0.4, 0.3, 0.3]
        overall_efficiency = (
            weights[0] * avg_resource_utilization +
            weights[1] * energy_efficiency +
            weights[2] * load_balance_score
        )

        return overall_efficiency, avg_resource_utilization, energy_efficiency, load_balance_score

    def add_vm_request(self, vm: VirtualMachine) -> bool:
        self.vms[vm.vm_id] = vm

        tx = Transaction(
            tx_id=f"tx_{uuid.uuid4().hex}",
            vm_id=vm.vm_id,
            node_id="",
            action="request",
            timestamp=time.time(),
            requester=vm.owner,
            gas_fee=random.uniform(0.01, 0.1)
        )

        self.blockchain.add_transaction(tx)
        return True

    def optimize_placement(self) -> Dict[str, str]:
        active_vms = list(self.vms.values())

        for node in self.nodes:
            node.cpu_used = node.memory_used = 0.0
            node.storage_used = node.network_used = 0.0

        new_placement = self.optimizer.nash_equilibrium_placement(active_vms)

        new_efficiency, new_resource_util, new_energy_eff, new_load_balance = \
            self.calculate_placement_efficiency(new_placement)

        best_previous_decision = self.blockchain.get_best_placement_from_ledger()

        final_placement = new_placement
        final_efficiency = new_efficiency
        final_resource_util = new_resource_util
        final_energy_eff = new_energy_eff
        final_load_balance = new_load_balance

        if best_previous_decision:
            if new_efficiency < best_previous_decision.overall_efficiency:
                final_placement = best_previous_decision.placement
                final_efficiency = best_previous_decision.overall_efficiency
                final_resource_util = best_previous_decision.resource_utilization
                final_energy_eff = best_previous_decision.energy_efficiency
                final_load_balance = best_previous_decision.load_balance_score

                for node in self.nodes:
                    node.cpu_used = node.memory_used = 0.0
                    node.storage_used = node.network_used = 0.0

                for vm_id, node_id in final_placement.items():
                    vm = self.vms.get(vm_id)
                    node = next((n for n in self.nodes if n.node_id == node_id), None)
                    if vm and node:
                        node.allocate_vm(vm)

        for vm_id, node_id in final_placement.items():
            tx = Transaction(
                tx_id=f"tx_{uuid.uuid4().hex}",
                vm_id=vm_id,
                node_id=node_id,
                action="allocate",
                timestamp=time.time(),
                requester="system",
                gas_fee=0.05,
                efficiency_score=final_efficiency
            )
            self.blockchain.add_transaction(tx)

        self.placement_map = final_placement
        return final_placement

    def mine_block(self) -> Optional[Block]:
        if not self.placement_map:
            return self.blockchain.mine_pending_transactions()

        overall_eff, resource_util, energy_eff, load_balance = \
            self.calculate_placement_efficiency(self.placement_map)

        decision = PlacementDecision(
            decision_id=str(len(self.blockchain.chain)),
            placement=self.placement_map.copy(),
            timestamp=time.time(),
            overall_efficiency=overall_eff,
            resource_utilization=resource_util,
            energy_efficiency=energy_eff,
            load_balance_score=load_balance
        )

        return self.blockchain.mine_pending_transactions(placement_decision=decision)

    def get_system_stats(self) -> Dict:
        total_vms = len(self.vms)
        placed_vms = len(self.placement_map)

        node_stats = []
        for node in self.nodes:
            usage = node.get_usage_vector()
            capacity = node.get_capacity_vector()
            utilization = np.divide(usage, capacity, out=np.zeros_like(usage), where=capacity != 0)

            node_stats.append({
                'node_id': node.node_id,
                'cpu_utilization': float(utilization[0]),
                'memory_utilization': float(utilization[1]),
                'storage_utilization': float(utilization[2]),
                'network_utilization': float(utilization[3])
            })

        return {
            'total_vms': total_vms,
            'placed_vms': placed_vms,
            'blockchain_blocks': len(self.blockchain.chain),
            'pending_transactions': len(self.blockchain.pending_transactions),
            'node_statistics': node_stats
        }

def create_sample_data():
    nodes = [
        PhysicalNode("node_1", 16.0, 32.0, 1000.0, 10.0, 0.15, "DC1"),
        PhysicalNode("node_2", 24.0, 48.0, 2000.0, 15.0, 0.18, "DC1"),
        PhysicalNode("node_3", 32.0, 64.0, 3000.0, 20.0, 0.20, "DC2"),
        PhysicalNode("node_4", 20.0, 40.0, 1500.0, 12.0, 0.16, "DC2")
    ]

    vms = [
        VirtualMachine("vm_1", 4.0, 8.0, 100.0, 2.0, 5, "user_a"),
        VirtualMachine("vm_2", 6.0, 12.0, 200.0, 3.0, 3, "user_b"),
        VirtualMachine("vm_3", 8.0, 16.0, 300.0, 4.0, 4, "user_c"),
        VirtualMachine("vm_4", 2.0, 4.0, 50.0, 1.0, 2, "user_d"),
        VirtualMachine("vm_5", 10.0, 20.0, 400.0, 5.0, 1, "user_e")
    ]

    return nodes, vms

def run_demonstration():
    nodes, vms = create_sample_data()
    system = VMPlacementSystem(nodes)

    for vm in vms:
        system.add_vm_request(vm)

    placement = system.optimize_placement()
    block = system.mine_block()

    stats = system.get_system_stats()
    print(f"Blocks: {len(system.blockchain.chain)}, Pending txs: {len(system.blockchain.pending_transactions)}")
    return system

if __name__ == "__main__":
    run_demonstration()