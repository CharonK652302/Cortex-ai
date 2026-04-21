"""
DSA Practice — 25 Problems for AI/ML Engineer Interviews
=========================================================
Week 1: Arrays + Strings (problems 1-8)
Week 2: HashMaps + Sliding Window (problems 9-15)
Week 3: Trees + Recursion (problems 16-20)
Week 4: AI-specific coding (problems 21-25)

Strategy:
  - Read problem, try 20 mins, then read solution
  - Understand WHY the approach works
  - Run each function — all have test cases

Run all:  python dsa_practice.py
Run one:  python dsa_practice.py 5
"""

import sys
from typing import Optional, List
from collections import defaultdict, Counter
import heapq
import math


# ═══════════════════════════════════════════════════════════════
# WEEK 1 — ARRAYS + STRINGS
# ═══════════════════════════════════════════════════════════════

def p1_two_sum(nums: list, target: int) -> list:
    """
    #1 Two Sum (Easy) — LeetCode 1
    Pattern: Hash map for O(1) lookup
    Time: O(n) | Space: O(n)
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def test_p1():
    assert p1_two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert p1_two_sum([3, 2, 4], 6) == [1, 2]
    print("P1 Two Sum ✓")


def p2_valid_anagram(s: str, t: str) -> bool:
    """
    #2 Valid Anagram (Easy) — LeetCode 242
    Pattern: Character frequency count
    Time: O(n) | Space: O(1)
    """
    if len(s) != len(t):
        return False
    return Counter(s) == Counter(t)

def test_p2():
    assert p2_valid_anagram("anagram", "nagaram") == True
    assert p2_valid_anagram("rat", "car") == False
    print("P2 Valid Anagram ✓")


def p3_contains_duplicate(nums: list) -> bool:
    """
    #3 Contains Duplicate (Easy) — LeetCode 217
    Pattern: Set for O(1) lookup
    Time: O(n) | Space: O(n)
    """
    return len(nums) != len(set(nums))

def test_p3():
    assert p3_contains_duplicate([1, 2, 3, 1]) == True
    assert p3_contains_duplicate([1, 2, 3, 4]) == False
    print("P3 Contains Duplicate ✓")


def p4_best_time_to_buy(prices: list) -> int:
    """
    #4 Best Time to Buy & Sell Stock (Easy) — LeetCode 121
    Pattern: Track min price, max profit greedily
    Time: O(n) | Space: O(1)
    """
    min_price = float("inf")
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

def test_p4():
    assert p4_best_time_to_buy([7, 1, 5, 3, 6, 4]) == 5
    assert p4_best_time_to_buy([7, 6, 4, 3, 1]) == 0
    print("P4 Best Time to Buy ✓")


def p5_max_subarray(nums: list) -> int:
    """
    #5 Maximum Subarray (Easy) — LeetCode 53
    Pattern: Kadane's algorithm — extend or restart
    Time: O(n) | Space: O(1)
    """
    current = max_sum = nums[0]
    for num in nums[1:]:
        current = max(num, current + num)
        max_sum = max(max_sum, current)
    return max_sum

def test_p5():
    assert p5_max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert p5_max_subarray([1]) == 1
    print("P5 Maximum Subarray ✓")


def p6_valid_palindrome(s: str) -> bool:
    """
    #6 Valid Palindrome (Easy) — LeetCode 125
    Pattern: Two pointers
    Time: O(n) | Space: O(1)
    """
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

def test_p6():
    assert p6_valid_palindrome("A man, a plan, a canal: Panama") == True
    assert p6_valid_palindrome("race a car") == False
    print("P6 Valid Palindrome ✓")


def p7_reverse_string(s: list) -> None:
    """
    #7 Reverse String (Easy) — LeetCode 344
    Pattern: Two pointers in-place
    Time: O(n) | Space: O(1)
    """
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

def test_p7():
    s = ["h", "e", "l", "l", "o"]
    p7_reverse_string(s)
    assert s == ["o", "l", "l", "e", "h"]
    print("P7 Reverse String ✓")


def p8_merge_sorted_arrays(nums1: list, m: int, nums2: list, n: int) -> None:
    """
    #8 Merge Sorted Array (Easy) — LeetCode 88
    Pattern: Fill from the end, two pointers
    Time: O(m+n) | Space: O(1)
    """
    p1, p2, p = m - 1, n - 1, m + n - 1
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]; p1 -= 1
        else:
            nums1[p] = nums2[p2]; p2 -= 1
        p -= 1
    nums1[:p2 + 1] = nums2[:p2 + 1]

def test_p8():
    nums1 = [1, 2, 3, 0, 0, 0]
    p8_merge_sorted_arrays(nums1, 3, [2, 5, 6], 3)
    assert nums1 == [1, 2, 2, 3, 5, 6]
    print("P8 Merge Sorted Arrays ✓")


# ═══════════════════════════════════════════════════════════════
# WEEK 2 — HASHMAPS + SLIDING WINDOW
# ═══════════════════════════════════════════════════════════════

def p9_group_anagrams(strs: list) -> list:
    """
    #9 Group Anagrams (Medium) — LeetCode 49
    Pattern: Sort each string as hash key
    Time: O(n * k log k) | Space: O(n)
    """
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())

def test_p9():
    result = p9_group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    assert len(result) == 3
    print("P9 Group Anagrams ✓")


def p10_top_k_frequent(nums: list, k: int) -> list:
    """
    #10 Top K Frequent Elements (Medium) — LeetCode 347
    Pattern: Count + heap
    Time: O(n log k) | Space: O(n)
    """
    count = Counter(nums)
    return [num for num, _ in count.most_common(k)]

def test_p10():
    assert set(p10_top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == {1, 2}
    print("P10 Top K Frequent ✓")


def p11_longest_substring_no_repeat(s: str) -> int:
    """
    #11 Longest Substring Without Repeating Characters (Medium) — LC 3
    Pattern: Sliding window with set
    Time: O(n) | Space: O(n)
    """
    char_set = set()
    left = max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len

def test_p11():
    assert p11_longest_substring_no_repeat("abcabcbb") == 3
    assert p11_longest_substring_no_repeat("bbbbb") == 1
    print("P11 Longest Substring No Repeat ✓")


def p12_max_sum_subarray_k(nums: list, k: int) -> int:
    """
    #12 Maximum Sum Subarray of Size K (Easy)
    Pattern: Fixed sliding window
    Time: O(n) | Space: O(1)
    """
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

def test_p12():
    assert p12_max_sum_subarray_k([2, 1, 5, 1, 3, 2], 3) == 9
    print("P12 Max Sum Subarray K ✓")


def p13_valid_parentheses(s: str) -> bool:
    """
    #13 Valid Parentheses (Easy) — LeetCode 20
    Pattern: Stack
    Time: O(n) | Space: O(n)
    """
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    return not stack

def test_p13():
    assert p13_valid_parentheses("()[]{}") == True
    assert p13_valid_parentheses("(]") == False
    print("P13 Valid Parentheses ✓")


def p14_product_except_self(nums: list) -> list:
    """
    #14 Product of Array Except Self (Medium) — LeetCode 238
    Pattern: Prefix + suffix product without division
    Time: O(n) | Space: O(1) extra
    """
    n = len(nums)
    result = [1] * n
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    return result

def test_p14():
    assert p14_product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    print("P14 Product Except Self ✓")


def p15_minimum_window_substring(s: str, t: str) -> str:
    """
    #15 Minimum Window Substring (Hard) — LeetCode 76
    Pattern: Variable sliding window with frequency maps
    Time: O(n) | Space: O(n)
    """
    if not t or not s:
        return ""
    need = Counter(t)
    have, total_needed = 0, len(need)
    left = 0
    best = ""
    window = defaultdict(int)
    for right, char in enumerate(s):
        window[char] += 1
        if char in need and window[char] == need[char]:
            have += 1
        while have == total_needed:
            candidate = s[left:right + 1]
            if not best or len(candidate) < len(best):
                best = candidate
            window[s[left]] -= 1
            if s[left] in need and window[s[left]] < need[s[left]]:
                have -= 1
            left += 1
    return best

def test_p15():
    assert p15_minimum_window_substring("ADOBECODEBANC", "ABC") == "BANC"
    print("P15 Minimum Window Substring ✓")


# ═══════════════════════════════════════════════════════════════
# WEEK 3 — TREES + RECURSION
# ═══════════════════════════════════════════════════════════════

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def p16_max_depth_tree(root: Optional[TreeNode]) -> int:
    """
    #16 Maximum Depth of Binary Tree (Easy) — LeetCode 104
    Pattern: DFS recursion
    Time: O(n) | Space: O(h)
    """
    if not root:
        return 0
    return 1 + max(p16_max_depth_tree(root.left), p16_max_depth_tree(root.right))

def test_p16():
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    assert p16_max_depth_tree(root) == 3
    print("P16 Max Depth Tree ✓")


def p17_invert_binary_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    #17 Invert Binary Tree (Easy) — LeetCode 226
    Pattern: Swap children recursively
    Time: O(n) | Space: O(h)
    """
    if not root:
        return None
    root.left, root.right = p17_invert_binary_tree(root.right), p17_invert_binary_tree(root.left)
    return root

def test_p17():
    root = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(7, TreeNode(6), TreeNode(9)))
    inverted = p17_invert_binary_tree(root)
    assert inverted.left.val == 7
    assert inverted.right.val == 2
    print("P17 Invert Binary Tree ✓")


def p18_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    """
    #18 Same Tree (Easy) — LeetCode 100
    Pattern: Simultaneous DFS
    Time: O(n) | Space: O(h)
    """
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return p18_same_tree(p.left, q.left) and p18_same_tree(p.right, q.right)

def test_p18():
    t1 = TreeNode(1, TreeNode(2), TreeNode(3))
    t2 = TreeNode(1, TreeNode(2), TreeNode(3))
    assert p18_same_tree(t1, t2) == True
    print("P18 Same Tree ✓")


def p19_level_order_traversal(root: Optional[TreeNode]) -> list:
    """
    #19 Binary Tree Level Order Traversal (Medium) — LeetCode 102
    Pattern: BFS with queue
    Time: O(n) | Space: O(n)
    """
    if not root:
        return []
    from collections import deque
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result

def test_p19():
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    assert p19_level_order_traversal(root) == [[3], [9, 20], [15, 7]]
    print("P19 Level Order Traversal ✓")


def p20_is_valid_bst(root: Optional[TreeNode], lo=float("-inf"), hi=float("inf")) -> bool:
    """
    #20 Validate Binary Search Tree (Medium) — LeetCode 98
    Pattern: Pass valid range down recursively
    Time: O(n) | Space: O(h)
    """
    if not root:
        return True
    if root.val <= lo or root.val >= hi:
        return False
    return (p20_is_valid_bst(root.left, lo, root.val) and
            p20_is_valid_bst(root.right, root.val, hi))

def test_p20():
    valid = TreeNode(2, TreeNode(1), TreeNode(3))
    invalid = TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))
    assert p20_is_valid_bst(valid) == True
    assert p20_is_valid_bst(invalid) == False
    print("P20 Validate BST ✓")


# ═══════════════════════════════════════════════════════════════
# WEEK 4 — AI/ML SPECIFIC CODING
# ═══════════════════════════════════════════════════════════════

def p21_cosine_similarity(v1: list, v2: list) -> float:
    """
    #21 Cosine Similarity from scratch
    Core to RAG — measures similarity between embeddings.
    Used in: FAISS ranking, semantic search, re-ranking.
    Time: O(n) | Space: O(1)
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return round(dot / (mag1 * mag2), 6)

def test_p21():
    v1 = [1, 0, 0]
    v2 = [1, 0, 0]
    assert p21_cosine_similarity(v1, v2) == 1.0
    v3 = [0, 1, 0]
    assert p21_cosine_similarity(v1, v3) == 0.0
    print("P21 Cosine Similarity ✓")


def p22_simple_tokenizer(text: str) -> list:
    """
    #22 Simple Tokenizer (word-level + BPE concepts)
    Understand tokenization — asked in LLM interviews.
    Covers: lowercasing, punctuation split, special tokens.
    """
    import re
    text = text.lower()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    tokens = text.split()
    return ["[CLS]"] + tokens + ["[SEP]"]

def p22_build_vocab(corpus: list) -> dict:
    """Build word-to-id vocabulary from corpus"""
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for text in corpus:
        for token in p22_simple_tokenizer(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def p22_encode(text: str, vocab: dict, max_len: int = 16) -> list:
    """Encode text to token IDs with padding/truncation"""
    tokens = p22_simple_tokenizer(text)[:max_len]
    ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    ids += [vocab["[PAD]"]] * (max_len - len(ids))
    return ids

def test_p22():
    corpus = ["the cat sat on the mat", "the dog ran fast"]
    vocab = p22_build_vocab(corpus)
    assert "[CLS]" in vocab
    assert "cat" in vocab
    encoded = p22_encode("the cat sat", vocab, max_len=8)
    assert len(encoded) == 8
    print("P22 Tokenizer ✓")


def p23_knn_classifier(train_X: list, train_y: list, test_point: list, k: int = 3) -> int:
    """
    #23 K-Nearest Neighbors from scratch
    Core ML algorithm — asked in ML coding rounds.
    Uses cosine similarity for distance.
    Time: O(n) | Space: O(k)
    """
    distances = []
    for i, train_point in enumerate(train_X):
        sim = p21_cosine_similarity(test_point, train_point)
        distances.append((1 - sim, train_y[i]))  # distance = 1 - similarity

    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    return Counter(k_nearest).most_common(1)[0][0]

def test_p23():
    train_X = [[1,0,0], [0.9,0.1,0], [0,0,1], [0,0.1,0.9]]
    train_y = [0, 0, 1, 1]
    test = [0.95, 0.05, 0]
    assert p23_knn_classifier(train_X, train_y, test, k=3) == 0
    print("P23 KNN Classifier ✓")


def p24_sliding_window_attention(tokens: list, window_size: int) -> list:
    """
    #24 Sliding Window Attention (conceptual implementation)
    Illustrates local attention — common interview topic for LLM roles.
    Each token attends only to its local window.
    """
    n = len(tokens)
    attention_outputs = []
    for i in range(n):
        lo = max(0, i - window_size // 2)
        hi = min(n, i + window_size // 2 + 1)
        local_window = tokens[lo:hi]
        attended = sum(local_window) / len(local_window)
        attention_outputs.append(round(attended, 4))
    return attention_outputs

def test_p24():
    tokens = [1.0, 2.0, 3.0, 4.0, 5.0]
    output = p24_sliding_window_attention(tokens, window_size=3)
    assert len(output) == len(tokens)
    assert output[2] == round((1 + 2 + 3 + 4 + 5) / 5, 4)  # center, full window
    print("P24 Sliding Window Attention ✓")


def p25_top_k_retrieval(query: list, documents: list, k: int = 3) -> list:
    """
    #25 Top-K Document Retrieval (RAG core)
    This is literally what FAISS does — asked in RAG interviews.
    Returns top-k documents by cosine similarity.
    Time: O(n log k) using heap
    """
    scored = []
    for i, doc in enumerate(documents):
        score = p21_cosine_similarity(query, doc["embedding"])
        heapq.heappush(scored, (-score, i))

    results = []
    for _ in range(min(k, len(scored))):
        neg_score, idx = heapq.heappop(scored)
        results.append({
            "rank": len(results) + 1,
            "score": round(-neg_score, 4),
            "text": documents[idx]["text"],
            "index": idx
        })
    return results

def test_p25():
    query = [1.0, 0.0, 0.0]
    docs = [
        {"text": "About machine learning", "embedding": [0.9, 0.1, 0.0]},
        {"text": "About cooking", "embedding": [0.0, 0.0, 1.0]},
        {"text": "About deep learning", "embedding": [0.95, 0.05, 0.0]},
        {"text": "About sports", "embedding": [0.0, 1.0, 0.0]},
    ]
    results = p25_top_k_retrieval(query, docs, k=2)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert "learning" in results[0]["text"]
    print("P25 Top-K Retrieval ✓")


# ═══════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════

ALL_TESTS = [
    test_p1, test_p2, test_p3, test_p4, test_p5,
    test_p6, test_p7, test_p8,
    test_p9, test_p10, test_p11, test_p12, test_p13, test_p14, test_p15,
    test_p16, test_p17, test_p18, test_p19, test_p20,
    test_p21, test_p22, test_p23, test_p24, test_p25,
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prob_num = int(sys.argv[1])
        test_fn = ALL_TESTS[prob_num - 1]
        print(f"Running problem #{prob_num}...")
        test_fn()
    else:
        print("Running all 25 problems...\n")
        passed = 0
        for i, test_fn in enumerate(ALL_TESTS, 1):
            try:
                test_fn()
                passed += 1
            except AssertionError as e:
                print(f"P{i} FAILED: {e}")
            except Exception as e:
                print(f"P{i} ERROR: {e}")
        print(f"\n{passed}/{len(ALL_TESTS)} tests passed")
