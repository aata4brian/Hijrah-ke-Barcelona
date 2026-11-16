import unittest
from buat_pembagian import MEMBERS, assignments, half_video_blocks, GAME


class PartitionTests(unittest.TestCase):
    def test_exact_coverage_no_overlap_or_gaps(self):
        rows=assignments()
        actual=[i for r in rows for i in range(r['frame_awal_inklusif'],r['frame_akhir_inklusif']+1)]
        expected=[i for _,a,b in half_video_blocks(GAME) for i in range(a,b+1)]
        self.assertEqual(actual,expected)
        self.assertEqual(len(actual),158857)
        self.assertEqual(len(set(actual)),len(actual))

    def test_balanced_and_half_boundaries_preserved(self):
        rows=assignments()
        counts=[sum(r['jumlah_frame'] for r in rows if r['anggota']==m) for m in MEMBERS]
        self.assertLessEqual(max(counts)-min(counts),1)
        for r in rows:
            block=next((a,b) for h,a,b in half_video_blocks(GAME) if h==r['babak'])
            self.assertLessEqual(block[0],r['frame_awal_inklusif'])
            self.assertLessEqual(r['frame_akhir_inklusif'],block[1])


if __name__=='__main__':
    unittest.main()
