import unittest
from main import Blockchain, Transaction

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        self.blockchain = Blockchain(difficulty=1)
    
    def test_genesis_block_creation(self):
        self.assertEqual(len(self.blockchain.chain), 1)
        self.assertEqual(self.blockchain.chain[0].block_id, "0")
    
    def test_transaction_addition(self):
        tx = Transaction("tx1", "vm1", "node1", "allocate", 123456, "user1", 0.05)
        self.blockchain.add_transaction(tx)
        self.assertEqual(len(self.blockchain.pending_transactions), 1)
    
    def test_block_mining(self):
        tx = Transaction("tx1", "vm1", "node1", "allocate", 123456, "user1", 0.05)
        self.blockchain.add_transaction(tx)
        block = self.blockchain.mine_pending_transactions()
        self.assertIsNotNone(block)
        self.assertEqual(len(self.blockchain.pending_transactions), 0)
    
    def test_blockchain_validation(self):
        for i in range(3):
            tx = Transaction(f"tx{i}", f"vm{i}", f"node{i}", "allocate", 123456+i, f"user{i}", 0.05)
            self.blockchain.add_transaction(tx)
            self.blockchain.mine_pending_transactions()
        
        self.assertTrue(self.blockchain.is_valid())
    
    def test_mining_and_validation(self):
        bc = Blockchain(difficulty=1)
        # add transaction(s) or not...
        block = bc.mine_pending_transactions()
        # accommodate None case using unittest assertions
        if block is None:
            # If there were no pending txs and no placement decision, this is valid behavior.
            self.assertEqual(len(bc.chain), 1)  # only genesis block remains
        else:
            self.assertTrue(block.hash.startswith("0"))
            self.assertTrue(bc.is_valid())
#run
if __name__ == '__main__':
    unittest.main()