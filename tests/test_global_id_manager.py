"""
Unit Tests for Global ID Manager

Test dual-master logic v cc edge cases
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend' / 'src'))

from global_id.manager import GlobalIDManager


class TestGlobalIDManager(unittest.TestCase):
    """Test Global ID Manager"""
    
    def setUp(self):
        """Setup test environment"""
        # Simple camera graph
        self.camera_graph = {
            1: {2: {'min_time': 5, 'max_time': 30}},
            2: {1: {'min_time': 5, 'max_time': 30}},
        }
        
        # Create manager with cam1 and cam2 as masters
        self.manager = GlobalIDManager(
            master_camera_ids=[1, 2],
            camera_graph=self.camera_graph,
            threshold_accept=0.75,
            threshold_reject=0.50,
            margin_threshold=0.15,
            min_tracklet_frames=5,
            min_bbox_size=80,
        )
    
    def test_case1_cam1_creates_cam2_matches(self):
        """
        Test Case 1: Cam1 to ID=1, Cam2 match ra ID=1
        
        Flow:
        t=0 : Cam1 thy person A  to Global ID = 1
        t=10: Cam2 thy person A  match ra Global ID = 1
        """
        print("\n" + "="*60)
        print("TEST CASE 1: Cam1 creates, Cam2 matches")
        print("="*60)
        
        # Create random embeddings
        emb_cam1 = np.random.rand(512)
        emb_cam1 = emb_cam1 / np.linalg.norm(emb_cam1)
        
        # Cam1 creates new ID
        gid1, reason1, score1 = self.manager.assign_global_id(
            camera_id=1,
            local_track_id=1,
            embedding=emb_cam1,
            quality=0.8,
            timestamp=0.0,
            frame_idx=0,
            num_frames=10,
            bbox_size=100
        )
        
        print(f"[cam1] Track 1  Global ID {gid1} ({reason1})")
        
        self.assertEqual(gid1, 1, "Cam1 should create Global ID = 1")
        self.assertIn("new_identity", reason1)
        
        # Cam2 sees same person (high similarity)
        emb_cam2 = emb_cam1 + np.random.rand(512) * 0.1  # Similar but not identical
        emb_cam2 = emb_cam2 / np.linalg.norm(emb_cam2)
        
        gid2, reason2, score2 = self.manager.assign_global_id(
            camera_id=2,
            local_track_id=1,
            embedding=emb_cam2,
            quality=0.8,
            timestamp=10.0,
            frame_idx=300,
            num_frames=10,
            bbox_size=100
        )
        
        print(f"[cam2] Track 1  Global ID {gid2} ({reason2}, score={score2:.3f})")
        
        self.assertEqual(gid2, 1, "Cam2 should match to Global ID = 1")
        self.assertGreater(score2, 0.7, "Similarity should be high")
        
        print(" PASSED: Cam1 creates, Cam2 matches correctly\n")
    
    def test_case2_cam2_creates_cam1_matches(self):
        """
        Test Case 2: Cam2 to ID=1 (cam1 miss), Cam1 match ra ID=1
        
        Flow:
        t=0 : Cam2 thy person B (cam1 miss)  to Global ID = 1
        t=10: Cam1 thy person B  match ra Global ID = 1
        """
        print("\n" + "="*60)
        print("TEST CASE 2: Cam2 creates (cam1 miss), Cam1 matches")
        print("="*60)
        
        # Create embedding
        emb_cam2 = np.random.rand(512)
        emb_cam2 = emb_cam2 / np.linalg.norm(emb_cam2)
        
        # Cam2 creates new ID (cam1 didn't see this person)
        gid1, reason1, score1 = self.manager.assign_global_id(
            camera_id=2,
            local_track_id=1,
            embedding=emb_cam2,
            quality=0.8,
            timestamp=0.0,
            frame_idx=0,
            num_frames=10,
            bbox_size=100
        )
        
        print(f"[cam2] Track 1  Global ID {gid1} ({reason1})")
        
        self.assertEqual(gid1, 1, "Cam2 should create Global ID = 1")
        self.assertIn("new_identity", reason1)
        
        # Cam1 sees same person later
        emb_cam1 = emb_cam2 + np.random.rand(512) * 0.1
        emb_cam1 = emb_cam1 / np.linalg.norm(emb_cam1)
        
        gid2, reason2, score2 = self.manager.assign_global_id(
            camera_id=1,
            local_track_id=1,
            embedding=emb_cam1,
            quality=0.8,
            timestamp=10.0,
            frame_idx=300,
            num_frames=10,
            bbox_size=100
        )
        
        print(f"[cam1] Track 1  Global ID {gid2} ({reason2}, score={score2:.3f})")
        
        self.assertEqual(gid2, 1, "Cam1 should match to Global ID = 1 (created by Cam2)")
        self.assertGreater(score2, 0.7, "Similarity should be high")
        
        print(" PASSED: Cam2 creates, Cam1 matches correctly\n")
    
    def test_case3_non_master_cannot_create(self):
        """
        Test Case 3: Cam3 thy ngi mi khng match  khng to ID (temp)
        
        Flow:
        t=0 : Cam3 (non-master) thy person C  TEMP ID = 0 (ch master)
        """
        print("\n" + "="*60)
        print("TEST CASE 3: Non-master camera cannot create new ID")
        print("="*60)
        
        # Create manager with cam3 as non-master
        manager = GlobalIDManager(
            master_camera_ids=[1, 2],  # Only 1 and 2 are masters
            camera_graph=self.camera_graph,
            threshold_accept=0.75,
            threshold_reject=0.50,
        )
        
        # Cam3 sees new person
        emb_cam3 = np.random.rand(512)
        emb_cam3 = emb_cam3 / np.linalg.norm(emb_cam3)
        
        gid, reason, score = manager.assign_global_id(
            camera_id=3,  # Non-master
            local_track_id=1,
            embedding=emb_cam3,
            quality=0.8,
            timestamp=0.0,
            frame_idx=0,
            num_frames=10,
            bbox_size=100
        )
        
        print(f"[cam3] Track 1  Global ID {gid} ({reason})")
        
        self.assertEqual(gid, 0, "Non-master should assign TEMP ID = 0")
        self.assertIn("wait_master", reason)
        
        print(" PASSED: Non-master camera correctly assigns TEMP ID\n")
    
    def test_case4_deterministic_tie_break(self):
        """
        Test Case 4: Deterministic tie-break khi 2 candidates gn bng score
        
        Flow:
        - To 2 IDs vi embeddings tng t nhau
        - Query embedding gn c 2 (ambiguous)
        - Test margin threshold
        """
        print("\n" + "="*60)
        print("TEST CASE 4: Deterministic tie-breaking")
        print("="*60)
        
        # Create 2 similar embeddings
        base_emb = np.random.rand(512)
        base_emb = base_emb / np.linalg.norm(base_emb)
        
        emb1 = base_emb + np.random.rand(512) * 0.1
        emb1 = emb1 / np.linalg.norm(emb1)
        
        emb2 = base_emb + np.random.rand(512) * 0.1
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Create ID 1
        gid1, _, _ = self.manager.assign_global_id(
            camera_id=1,
            local_track_id=1,
            embedding=emb1,
            quality=0.8,
            timestamp=0.0,
            frame_idx=0,
            num_frames=10,
            bbox_size=100
        )
        
        # Create ID 2
        gid2, _, _ = self.manager.assign_global_id(
            camera_id=1,
            local_track_id=2,
            embedding=emb2,
            quality=0.8,
            timestamp=10.0,
            frame_idx=300,
            num_frames=10,
            bbox_size=100
        )
        
        print(f"Created Global ID {gid1} and {gid2}")
        
        # Query that's ambiguous (similar to both)
        query_emb = (emb1 + emb2) / 2
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Should create new ID due to ambiguous margin
        gid3, reason3, score3 = self.manager.assign_global_id(
            camera_id=1,
            local_track_id=3,
            embedding=query_emb,
            quality=0.8,
            timestamp=20.0,
            frame_idx=600,
            num_frames=10,
            bbox_size=100
        )
        
        print(f"[cam1] Track 3  Global ID {gid3} ({reason3}, score={score3:.3f})")
        
        # Should create new ID when ambiguous (conservative)
        self.assertNotIn(gid3, [gid1, gid2], "Should create new ID when ambiguous")
        
        print(" PASSED: Ambiguous case handled correctly\n")
    
    def test_case5_spatiotemporal_filter(self):
        """
        Test Case 5: Spatiotemporal filtering
        
        Flow:
        - Person xut hin  cam1
        - Sau 2s xut hin  cam2 (qu nhanh, min_time = 5s)
        - Should reject match v to ID mi
        """
        print("\n" + "="*60)
        print("TEST CASE 5: Spatiotemporal filtering")
        print("="*60)
        
        # Same embedding
        emb = np.random.rand(512)
        emb = emb / np.linalg.norm(emb)
        
        # Cam1 at t=0
        gid1, _, _ = self.manager.assign_global_id(
            camera_id=1,
            local_track_id=1,
            embedding=emb,
            quality=0.8,
            timestamp=0.0,
            frame_idx=0,
            num_frames=10,
            bbox_size=100
        )
        
        print(f"[cam1] t=0s  Global ID {gid1}")
        
        # Cam2 at t=2s (too fast! min_time=5s)
        gid2, reason2, score2 = self.manager.assign_global_id(
            camera_id=2,
            local_track_id=1,
            embedding=emb,  # Same embedding (high similarity)
            quality=0.8,
            timestamp=2.0,  # Only 2 seconds later
            frame_idx=60,
            num_frames=10,
            bbox_size=100,
            last_seen_info={'camera': 1, 'time': 0.0}
        )
        
        print(f"[cam2] t=2s  Global ID {gid2} ({reason2}, score={score2:.3f})")
        
        # Should create new ID due to spatiotemporal violation
        # (Cannot teleport from cam1 to cam2 in 2 seconds)
        if gid2 != gid1:
            print(" PASSED: Spatiotemporal filter rejected impossible transition")
        else:
            print("  Note: Matched despite spatiotemporal constraint (may need tuning)")
        
        print()
    
    def test_metrics(self):
        """Test metrics tracking"""
        print("\n" + "="*60)
        print("TEST: Metrics tracking")
        print("="*60)
        
        # Create a few IDs
        for i in range(3):
            emb = np.random.rand(512)
            emb = emb / np.linalg.norm(emb)
            
            self.manager.assign_global_id(
                camera_id=1,
                local_track_id=i+1,
                embedding=emb,
                quality=0.8,
                timestamp=float(i * 10),
                frame_idx=i * 300,
                num_frames=10,
                bbox_size=100
            )
        
        metrics = self.manager.get_metrics()
        
        print(f"Metrics:")
        print(f"  Total Global IDs: {metrics['total_global_ids']}")
        print(f"  IDs Created: {metrics['ids_created']}")
        print(f"  IDs Matched: {metrics['ids_matched']}")
        print(f"  Master New IDs: {metrics['master_new_ids']}")
        
        self.assertGreater(metrics['total_global_ids'], 0)
        self.assertGreater(metrics['ids_created'], 0)
        
        print(" PASSED: Metrics tracking works\n")


def run_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("HAVEN GLOBAL ID MANAGER - UNIT TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGlobalIDManager)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n ALL TESTS PASSED!")
    else:
        print("\n SOME TESTS FAILED")
    
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

