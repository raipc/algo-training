package org.raipc.leetcode

import java.util.*
import kotlin.collections.ArrayList

class Solution {
    // 1162. As Far from Land as Possible
    fun maxDistance(grid: Array<IntArray>): Int {
        val dimension = grid.size
        val directions = arrayOf(intArrayOf(-1, 0), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(0, 1))
        val queue = ArrayDeque<IntArray>()
        for (i in 0 until dimension) {
            for (j in 0 until dimension) {
                if (grid[i][j] == 1) {
                    queue += intArrayOf(i, j)
                }
            }
        }
        if (queue.isEmpty() || queue.size == dimension * dimension) {
            return -1
        }
        var iterations = -1
        while (!queue.isEmpty()) {
            iterations += 1
            repeat(queue.size) {
                val (i, j) = queue.removeFirst()
                for (direction in directions) {
                    val (x, y) = intArrayOf(i + direction[0], j + direction[1])
                    if (x in 0 until dimension && y in 0 until dimension && grid[x][y] == 0) {
                        grid[x][y] = 1
                        queue += intArrayOf(x, y)
                    }
                }
            }
        }
        return iterations
    }

    // 1480. Running Sum of 1d Array
    fun runningSum(nums: IntArray): IntArray {
        var sum = 0
        return IntArray(nums.size) {
            sum += nums[it]
            sum
        }
    }

    // 724. Find Pivot Index
    fun pivotIndex(nums: IntArray): Int {
        var leftSum = 0
        var rightSum = nums.sum()
        var pivotElement = 0
        for (i in nums.indices) {
            leftSum += pivotElement
            pivotElement = nums[i]
            rightSum -= pivotElement
            if (leftSum == rightSum) {
                return i
            }
        }
        return -1
    }

    // 205. Isomorphic Strings
    fun isIsomorphic(s: String, t: String): Boolean {
        val fromSToT = hashMapOf<Int, Int>()
        val fromTToS = hashMapOf<Int, Int>()
        for (i in s.indices) {
            val sCode = s[i].toInt()
            val tCode = t[i].toInt()
            fromSToT.put(sCode, tCode)?.also { if (it != tCode) return false }
            fromTToS.put(tCode, sCode)?.also { if (it != sCode) return false }
        }
        return true
    }

    // 392. Is Subsequence
    fun isSubsequence(s: String, t: String): Boolean {
        if (s.isEmpty()) {
            return true
        }
        var targetIndex = 0
        var targetCharacter = s[targetIndex]
        for (ch in t) {
            if (ch == targetCharacter) {
                ++targetIndex
                if (targetIndex == s.length) {
                    return true
                }
                targetCharacter = s[targetIndex]
            }
        }
        return false
    }

    // 393. UTF-8 Validation
    fun validUtf8(data: IntArray): Boolean {
        var expectBytes = 0
        for (byte in data) {
            if (expectBytes == 0) {
                expectBytes = when {
                    (byte and 0b1000_0000) == 0 -> 0
                    (byte and 0b1110_0000) == 0b1100_0000 -> 1
                    (byte and 0b1111_0000) == 0b1110_0000 -> 2
                    (byte and 0b1111_1000) == 0b1111_0000 -> 3
                    else -> return false
                }
            } else {
                if ((byte and 0b1100_0000) != 0b1000_0000) {
                    return false
                }
                expectBytes--
            }
        }
        return expectBytes == 0
    }

    // 4. Median of Two Sorted Arrays
    fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
        val (first, second) = if (nums1.size <= nums2.size) arrayOf(nums1, nums2) else arrayOf(nums2, nums1)
        val leftSize: Int = first.size
        val rightSize: Int = second.size

        var start = 0
        var end = leftSize
        while (start <= end) {
            val partition1 = (start + end) / 2
            val partition2 = (leftSize + rightSize + 1) / 2 - partition1
            val maxLeftNums1 = if (partition1 == 0) Int.MIN_VALUE else first[partition1 - 1]
            val minRightNums1 = if (partition1 == leftSize) Int.MAX_VALUE else first[partition1]
            val maxLeftNums2 = if (partition2 == 0) Int.MIN_VALUE else second[partition2 - 1]
            val minRightNums2 = if (partition2 == rightSize) Int.MAX_VALUE else second[partition2]
            if (maxLeftNums1 <= minRightNums2 && maxLeftNums2 <= minRightNums1) {
                return if ((leftSize + rightSize) % 2 == 0) {
                    (Math.max(maxLeftNums1, maxLeftNums2) + Math.min(minRightNums1, minRightNums2)) / 2.0
                } else {
                    Math.max(maxLeftNums1, maxLeftNums2).toDouble()
                }
            } else if (maxLeftNums1 > minRightNums2) {
                end = partition1 - 1
            } else {
                start = partition1 + 1
            }
        }
        return Double.NaN
    }

    // 704. Binary Search
    fun search(nums: IntArray, target: Int): Int {
        var low = 0
        var high = nums.size - 1
        while (low <= high) {
            val mid = (low + high) / 2
            val midVal = nums[mid]
            when {
                midVal < target -> low = mid + 1
                midVal > target -> high = mid - 1
                else -> return mid
            }
        }
        return -1
    }

    // 35. Search Insert Position
    fun searchInsert(nums: IntArray, target: Int): Int {
        var low = 0
        var high = nums.size - 1
        while (low <= high) {
            val mid = (low + high) / 2
            val midVal = nums[mid]
            when {
                midVal < target -> low = mid + 1
                midVal > target -> high = mid - 1
                else -> return mid
            }
        }
        return low
    }

    // 14. Longest Common Prefix
    fun longestCommonPrefix(strs: Array<String>): String {
        if (strs.size == 1) return strs[0]
        val result = StringBuilder()
        val minSize = strs.map { it.length }.min()!!
        for (i in 0 until minSize) {
            val ch = strs[0][i]
            if ((1 until strs.size).all { strs[it][i] == ch }) result.append(ch) else break
        }
        return result.toString()
    }

    // 13. Roman to Integer
    fun romanToInt(s: String): Int {
        var accumulator = 0
        var prevValue = 0
        for (symbol in s) {
            val value = when(symbol) {
                'I' -> 1
                'V' -> 5
                'X' -> 10
                'L' -> 50
                'C' -> 100
                'D' -> 500
                'M' -> 1000
                else -> throw IllegalArgumentException()
            }
            accumulator += value
            if (value > prevValue) {
                accumulator -= 2 * prevValue
            }
            prevValue = value
        }
        return accumulator
    }

    // 189. Rotate Array
    fun rotate(nums: IntArray, k: Int): IntArray {
        fun reverse(from: Int, to: Int) {
            for (i in from until from + (to - from) / 2) {
                val rightIdx = to-i+from-1
                val tmp = nums[i]
                nums[i] = nums[rightIdx]
                nums[rightIdx] = tmp
            }
        }
        val size = nums.size
        val normalizedRotations = k % size
        if (normalizedRotations > 0) {
            reverse(0, size)
            reverse(0, normalizedRotations)
            reverse(normalizedRotations, size)
        }
        return nums
    }


    // 977. Squares of a Sorted Array
    fun sortedSquares(nums: IntArray): IntArray {
        val size = nums.size
        val idxOfNonNegative = nums.indexOfFirst { it >= 0 }
        var cnt = 0
        val result = IntArray(size)
        var positivesIdx = if (idxOfNonNegative >= 0) idxOfNonNegative else size
        var negativesIdx = positivesIdx - 1
        while (negativesIdx >= 0 && positivesIdx < size) {
            if (nums[positivesIdx] < -nums[negativesIdx]) {
                result[cnt] += nums[positivesIdx] * nums[positivesIdx]
                positivesIdx++
            } else {
                result[cnt] += nums[negativesIdx] * nums[negativesIdx]
                negativesIdx--
            }
            cnt++
        }
        for (i in positivesIdx until  size) {
            result[cnt] += nums[i] * nums[i]
            ++cnt
        }
        for (i in negativesIdx downTo 0) {
            result[cnt] += nums[i] * nums[i]
            ++cnt
        }

        return result
    }

    // 2477. Minimum Fuel Cost to Report to the Capital
    fun minimumFuelCost(roads: Array<IntArray>, seats: Int): Long {
        val connectedCities = Array(roads.size + 1) { mutableListOf<Int>() }
        roads.forEach {(from, to) -> connectedCities[from].add(to); connectedCities[to].add(from) }
        var fuelSpent = 0L
        fun dfs(location: Int, previous: Int): Int = (1 + connectedCities[location].sumBy { if (it != previous) dfs(it, location) else 0 })
            .also { passengers -> if (location != 0) fuelSpent += Math.ceil(passengers.toDouble() / seats).toLong() }
        dfs(0, -1)
        return fuelSpent
    }

    // 21. Merge Two Sorted Lists
    fun mergeTwoLists(list1: ListNode?, list2: ListNode?): ListNode? {
        if (list1 == null) return list2
        if (list2 == null) return list1
        var (iter1, iter2: ListNode?, resultRoot) = if (list1.`val` <= list2.`val`) {
            arrayOf(list1.next, list2, list1)
        } else {
            arrayOf(list1, list2.next, list2)
        }
        var resultIter: ListNode = resultRoot!!
        while (iter1 != null && iter2 != null) {
            val next: ListNode
            if (iter1.`val` <= iter2.`val`) {
                next = iter1
                iter1 = iter1.next
            } else {
                next = iter2
                iter2 = iter2.next
            }
            resultIter.next = next
            resultIter = next
        }
        if (iter1 != null) {
            resultIter.next = iter1
        } else if (iter2 != null) {
            resultIter.next = iter2
        }
        return resultRoot
    }

    // 206. Reverse Linked List
    fun reverseList(head: ListNode?): ListNode? {
        if (head == null) return null
        var prevPrev: ListNode? = null
        var prev = head

        while (prev != null) {
            val cur = prev.next
            prev.next = prevPrev
            prevPrev = prev
            prev = cur
        }
        return prevPrev
    }

    // 20. Valid Parentheses
    fun isValid(s: String): Boolean {
        val stack = ArrayDeque<Char>()
        for (ch in s) {
            when(ch) {
                '(', '[', '{' -> stack.addLast(ch)
                ')' -> stack.pollLast()?.takeIf { it == '(' } ?: return false
                ']' -> stack.pollLast()?.takeIf { it == '[' } ?: return false
                '}' -> stack.pollLast()?.takeIf { it == '{' } ?: return false
                else -> throw IllegalArgumentException()
            }
        }
        return stack.isEmpty()
    }

    // 6. Zigzag Conversion
    fun convert(s: String, numRows: Int): String = if (numRows == 1) s else buildString {
        val period = 2 * numRows - 2
        for (row in 0 until numRows) {
            for (i in row until s.length step period) {
                append(s[i])
                if (row > 0 && row < numRows - 1 && i + period - 2 * row < s.length) {
                    append(s[i+period-2*row])
                }
            }
        }
    }

    // 1523. Count Odd Numbers in an Interval Range
    fun countOdds(low: Int, high: Int): Int =
        (high - low) / 2 + if (low % 2 == 1 || high % 2 == 1) 1 else 0

    // 283. Move Zeroes
    fun moveZeroes(nums: IntArray): Unit {
        var shift = 0
        for (i in nums.indices) {
            val el = nums[i]
            if (el == 0) {
                ++shift
            } else if (shift > 0) {
                nums[i-shift] = el
            }
        }
        for (i in nums.size - shift until nums.size) {
            nums[i] = 0
        }
    }

    // 167. Two Sum II - Input Array Is Sorted
    fun twoSum(numbers: IntArray, target: Int): IntArray {
        var leftIdx = 0
        var rightIdx = numbers.size - 1
        while (true) {
            val left = numbers[leftIdx]
            val right = numbers[rightIdx]
            when {
                left + right > target -> --rightIdx
                left + right < target -> ++leftIdx
                else -> return intArrayOf(leftIdx + 1, rightIdx + 1)

            }
        }
    }

    // 876. Middle of the Linked List
    fun middleNode(head: ListNode?): ListNode? {
        var slow = head
        var fast = head
        while (fast != null) {
            fast = fast.next
            if (fast != null) slow = slow?.next
            fast = fast?.next

        }
        return slow
    }

    // 142. Linked List Cycle II
    fun detectCycle(head: ListNode?): ListNode? {
        var fast = head
        var slow = head
        do {
            slow = slow?.next
            fast = fast?.next?.next
        } while (fast != null && slow != fast)
        if (fast == null) return null
        var cycleStart = head
        while (cycleStart !== slow) {
            cycleStart = cycleStart!!.next
            slow = slow!!.next
        }
        return cycleStart
    }

    // 23. Merge k Sorted Lists
    fun mergeKLists(lists: Array<ListNode?>): ListNode? {
        if (lists.isEmpty()) return null
        val pq = PriorityQueue<ListNode>(lists.size) { a, b -> a.`val`.compareTo(b.`val`) }
        lists.forEach { if (it != null) pq += it }
        if (pq.isEmpty()) return null
        val root = pq.poll()!!
        root.next?.let { pq += it }
        var iter: ListNode? = root
        while (!pq.isEmpty()) {
            pq.poll()!!.let {
                iter!!.next = it
                iter = it
                it.next?.let { next -> pq += next }
            }
        }
        return root
    }

    // 9. Palindrome Number
    fun isPalindrome(x: Int): Boolean {
        if (x == 0) return true
        if (x < 0 || x % 10 == 0) return false
        var rest = x
        var reverted = 0
        while (rest > reverted) {
            reverted = reverted * 10 + rest % 10;
            rest /= 10
        }
        return rest == reverted || rest == reverted / 10
    }

    // 8. String to Integer (atoi)
    fun myAtoi(s: String): Int {
        val len = s.length
        var i = 0
        while (i < len && s[i] == ' ') ++i
        var accumulator = 0L
        var mult = 1
        if (i < len) {
            when(s[i]) {
                '-' -> {++i; mult = -1}
                '+' -> ++i
            }
            val greaterThanMoxPositiveInt = Integer.MAX_VALUE + 1L
            while (i < len && accumulator <= greaterThanMoxPositiveInt) {
                val ch = s[i]
                if (ch in '0'..'9') {
                    accumulator = accumulator * 10 + (ch - '0')
                } else break
                ++i
            }
        }
        val result = mult * accumulator
        return when {
            result >= Integer.MAX_VALUE -> Integer.MAX_VALUE
            result <= Integer.MIN_VALUE -> Integer.MIN_VALUE
            else -> result.toInt()
        }
    }

    // 10. Regular Expression Matching
    fun isMatch(s: String, p: String): Boolean {
        val dp = Array(s.length + 1) { BitSet(p.length + 1) }.apply { this[s.length][p.length] = true }
        for (i in s.length downTo 0) {
            for (j in p.length - 1 downTo 0) {
                val firstMatch = i < s.length && (p[j] == s[i] || p[j] == '.')
                dp[i][j] = if (j + 1 < p.length && p[j + 1] == '*') {
                    dp[i][j+2] || firstMatch && dp[i+1][j]
                } else {
                    firstMatch && dp[i+1][j+1]
                }
            }
        }
        return dp[0][0]
    }

    // 67. Add Binary
    fun addBinary(a: String, b: String): String {
        if (a.length > b.length) {
            return addBinary(b, a)
        }
        return buildString(maxOf(a.length, b.length) + 1) {
            var mem = 0
            val leftLength = a.length
            val rightLength = b.length
            val zeroCode = '0'.toInt()
            val zeroCodeDouble = zeroCode * 2
            for (i in 0 until leftLength) {
                val left = a[leftLength - i - 1]
                val right = b[rightLength - i - 1]
                val sum = left.toInt() + right.toInt() + mem - zeroCodeDouble
                mem = sum / 2
                append(((sum % 2) + zeroCode).toChar())
            }
            for (i in leftLength until rightLength) {
                val sum = b[rightLength - i - 1].toInt() + mem - zeroCode
                mem = sum / 2
                append(((sum % 2) + zeroCode).toChar())
            }
            if (mem > 0) {
                append('1')
            }
        }.reversed()
    }

    // 121. Best Time to Buy and Sell Stock
    fun maxProfit(prices: IntArray): Int {
        var maxPrice = -1
        var minPrice = -1
        var maxProfit = 0
        for (i in prices.size - 1 downTo 0) {
            prices[i].let {
                if (it > maxPrice)  {
                    maxProfit = maxOf(maxProfit, maxPrice - minPrice)
                    maxPrice = it
                    minPrice = it
                } else if (it < minPrice) {
                    minPrice = it
                }
            }
        }
        return maxOf(maxProfit, maxPrice - minPrice)
    }

    // 409. Longest Palindrome
    fun longestPalindrome(s: String): Int {
        val asciiTableSize = 127
        val counters = IntArray(asciiTableSize)
        for (ch in s) {
            counters[ch.toInt()] += 1
        }
        var oddSum = 0
        var evenHalfSum = 0
        for (i in 'A'.toInt() until asciiTableSize) {
            counters[i].let {
                evenHalfSum += (it / 2)
                if (oddSum == 0 && it % 2 == 1) {
                    oddSum = 1
                }
            }
        }
        return evenHalfSum * 2 + oddSum
    }

    // 344. Reverse String
    fun reverseString(s: CharArray) = s.reverse()


    // 557. Reverse Words in a String III
    fun reverseWords(s: String): String = s.splitToSequence(' ')
        .joinToString(" ") { it.reversed() }


    // 26. Remove Duplicates from Sorted Array
    fun removeDuplicates(nums: IntArray): Int {
        var prev = nums[0]
        var deleteCount = 0
        for (i in 1 until nums.size) {
            val value = nums[i]
            if (deleteCount > 0) {
                nums[i - deleteCount] = value
            }
            if (value == prev) {
                ++deleteCount
            }
            prev = value
        }
        return nums.size - deleteCount
    }

    // 989. Add to Array-Form of Integer
    fun addToArrayForm(num: IntArray, k: Int): List<Int> {
        var rest = k
        val result = mutableListOf<Int>()
        for (i in num.size - 1 downTo 0) {
            val sum = num[i] + (rest % 10)
            result.add(sum % 10)
            rest = rest / 10 + sum / 10
        }
        while (rest > 0) {
            result.add(rest % 10)
            rest /= 10
        }
        return result.asReversed()
    }

    // 589. N-ary Tree Preorder Traversal
    fun preorder(root: Node?): List<Int> {
        if (root == null) return listOf()
        val stack = ArrayDeque<Node>().apply { add(root) }
        val result = mutableListOf<Int>()
        while (!stack.isEmpty()) {
            val node = stack.pollLast()!!
            result += node.`val`
            for (i in node.children.size - 1 downTo 0) {
                stack += node.children[i]
            }
        }
        return result
    }

    // 102. Binary Tree Level Order Traversal
    fun levelOrder(root: TreeNode?): List<List<Int>> {
        if (root == null) return listOf()
        val result = mutableListOf<MutableList<Int>>()
        val stack = ArrayDeque<Pair<TreeNode, Int>>().apply { add(Pair(root, 0)) }
        while (stack.isNotEmpty()) {
            val (node, level) = stack.pollLast()!!
            if (result.size <= level) {
                result += mutableListOf<Int>()
            }
            result[level].add(node.`val`)
            node.right?.let { stack += Pair(it, level + 1) }
            node.left?.let { stack += Pair(it, level + 1) }
        }
        return result
    }

    // 19. Remove Nth Node From End of List
    fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
        fun countFromTailAndRemoveMatching(node: ListNode?): Int =
            if (node == null) 0 else 1 + countFromTailAndRemoveMatching(node.next)
                .also { if (it == n) { node.next = node.next?.next } }
        return if (head == null) null else ListNode(Int.MIN_VALUE).apply { next = head }
            .apply { countFromTailAndRemoveMatching(this) }
            .next
    }

    // 5. Longest Palindromic Substring
    fun longestPalindromicSubstring(s: String): String {
        val preprocessed = CharArray(s.length * 2 + 3)
            .apply { for (i in s.indices) this[2*i+2] = s[i] }
            .apply { this[0] = '^' }
            .apply { this[lastIndex] = '$' }
        val palLengths = IntArray(preprocessed.size)
        var left = 1
        var right = 1
        for (center in 1 ..  preprocessed.size - 2) {
            palLengths[center] = maxOf(0, minOf(right - center, palLengths[left + (right - center)]))
            while (preprocessed[center - palLengths[center]] ==
                   preprocessed[center + palLengths[center]]) {
                palLengths[center]++
            }
            if (center + palLengths[center] > right) {
                left = center - palLengths[center]
                right = center + palLengths[center]
            }
        }
        val maxIdx = palLengths.indices.maxBy { palLengths[it] } ?: 0
        return s.substring((maxIdx - palLengths[maxIdx]) / 2, (maxIdx + palLengths[maxIdx] - 2) / 2)
    }

    // 3. Longest Substring Without Repeating Characters
    fun lengthOfLongestSubstring(s: String): Int {
        var maxLength = 0
        var startIdx = 0
        val charCodesToIndex = mutableMapOf<Int, Int>()
        for (i in s.indices) {
            charCodesToIndex.put(s[i].toInt(), i)?.also { prevIdx ->
                if (prevIdx >= startIdx) {
                    maxLength = maxOf(maxLength, i - startIdx)
                    startIdx = prevIdx + 1
                }
            }
        }
        return maxOf(maxLength, s.length - startIdx)
    }

    fun reverseKGroup(head: ListNode?, k: Int): ListNode? {
        fun reverseFullGroup(head: ListNode?, groupSize: Int, out: Array<ListNode?>): Array<ListNode?> {
            var prev: ListNode? = null
            var cur = head
            var cnt = 0
            while (cur != null && cnt < groupSize) {
                val next = cur.next
                cur.next = prev
                prev = cur
                cur = next
                ++cnt
            }
            if (cnt < groupSize) {
                return reverseFullGroup(prev, cnt, out)
            }
            out[0] = prev
            out[1] = cur
            return out
        }
        var result: ListNode? = null
        var node = head
        val tmp = arrayOfNulls<ListNode?>(2)
        var prevTail: ListNode? = null
        while (node != null) {
            val (headOfGroupAfterReverse, nextNode) = reverseFullGroup(node, k, tmp)
            prevTail?.next = headOfGroupAfterReverse
            prevTail = node
            node = nextNode
            if (result == null) {
                result = headOfGroupAfterReverse
            }
        }
        return result
    }

    // 104. Maximum Depth of Binary Tree
    fun maxDepth(root: TreeNode?): Int {
        if (root == null) return 0
        val stack: Deque<TreeNode> = ArrayDeque<TreeNode>().apply { addLast(root) }
        var depth = 0
        do {
            ++depth
            for (i in 0 until stack.size) {
                val node = stack.pollFirst()!!
                node.left?.let { stack.addLast(it) }
                node.right?.let { stack.addLast(it) }
            }
        } while (!stack.isEmpty())
        return depth
    }

    fun maxDepthRecursive(root: TreeNode?): Int =
        if (root == null) 0 else maxOf(maxDepthRecursive(root.left), maxDepthRecursive(root.right)) + 1

    // 2006. Count Number of Pairs With Absolute Difference K
    fun countKDifference(nums: IntArray, k: Int): Int {
        val matchingIndexes = nums.indices.groupBy({ nums[it] }, { it })
        return nums.indices.sumBy { idx ->
            (matchingIndexes[nums[idx] - k]?.count { it > idx } ?: 0) +
            (matchingIndexes[nums[idx] + k]?.count { it > idx } ?: 0)
        }
    }

    fun countKDifferenceNaive(nums: IntArray, k: Int): Int {
        var count = 0
        for (i in 0 until nums.size - 1) {
            for (j in i + 1 until nums.size) {
                if (Math.abs(nums[i] - nums[j]) == k) {
                    ++count
                }
            }
        }
        return count
    }

    // 783. Minimum Distance Between BST Nodes
    fun minDiffInBST(root: TreeNode?): Int {
        if (root == null) return Int.MAX_VALUE
        val list = ArrayList<TreeNode>().apply { add(root) }
        var i = 0
        while (i < list.size) {
            val node = list[i]
            node.left?.let { list += it }
            node.right?.let { list += it }
            ++i
        }
        list.sortBy { it.`val` }
        return (0 until list.size - 1).minBy { list[it+1].`val` - list[it].`val` }!!
            .let { list[it+1].`val` - list[it].`val` }
    }

    // 530. Minimum Absolute Difference in BST
    fun getMinimumDifference(root: TreeNode?): Int {
        var minDiff = Int.MAX_VALUE
        var prev: TreeNode? = null
        fun traverseInOrder(node: TreeNode?): Int {
            if (node == null) return minDiff
            traverseInOrder(node.left)
            if (prev != null) {
                minDiff = minOf(minDiff, node.`val` - prev!!.`val`)
            }
            prev = node
            return traverseInOrder(node.right)
        }
        return traverseInOrder(root)
    }

    // 98. Validate Binary Search Tree
    fun isValidBST(root: TreeNode?): Boolean {
        fun isValidRecursive(node: TreeNode, rangeLeft: Long, rangeRight: Long): Boolean = node.`val`.let { value ->
            (value > rangeLeft) && (value < rangeRight) &&
            (node.left?.let { it.`val` < value && isValidRecursive(it, rangeLeft, value.toLong()) } ?: true) &&
            (node.right?.let { it.`val` > value && isValidRecursive(it, value.toLong(), rangeRight) } ?: true)
        }
        return root == null || isValidRecursive(root, Long.MIN_VALUE, Long.MAX_VALUE)
    }

    // 235. Lowest Common Ancestor of a Binary Search Tree
    fun lowestCommonAncestor(root: TreeNode?, p: TreeNode?, q: TreeNode?): TreeNode? {
        return when {
            root == p || root == q || (p!!.`val` - root!!.`val`).toLong() * (q!!.`val` - root.`val`) < 0L -> root
            p.`val` < root.`val` -> lowestCommonAncestor(root.left, p, q)
            else -> lowestCommonAncestor(root.right, p, q)
        }
    }

    // 733. Flood Fill
    fun floodFill(image: Array<IntArray>, sr: Int, sc: Int, color: Int): Array<IntArray> {
        val m = image.size
        val n = image[0].size
        val directions = arrayOf(intArrayOf(-1, 0), intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(0, -1))
        val prevColor = image[sr][sc]
        if (color != prevColor) {
            image[sr][sc] = color
            val queue = ArrayDeque<Pair<Int, Int>>().apply { add(Pair(sr, sc)) }
            while (!queue.isEmpty()) {
                val (x, y) = queue.removeFirst()
                for ((dx, dy) in directions) {
                    val i = x + dx
                    val j = y + dy
                    if (i in 0 until m && j in 0 until n && image[i][j] == prevColor) {
                        queue += Pair(i, j)
                        image[i][j] = color
                    }
                }
            }
        }
        return image
    }

    // 695. Max Area of Island
    fun maxAreaOfIsland(grid: Array<IntArray>): Int {
        val m = grid.size
        val n = grid[0].size
        var maxArea = 0
        val directions = arrayOf(intArrayOf(-1, 0), intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(0, -1))
        val queue = ArrayDeque<Pair<Int, Int>>()
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (grid[i][j] != 1) continue
                var currentComponentSize = 0
                queue += Pair(i, j)
                grid[i][j] = 0
                while(!queue.isEmpty()) {
                    val (x, y) = queue.removeFirst()
                    ++currentComponentSize
                    for ((dx, dy) in directions) {
                        val xx = x + dx
                        val yy = y + dy
                        if (xx in 0 until m && yy in 0 until n && grid[xx][yy] == 1) {
                            grid[xx][yy] = 0
                            queue += Pair(xx, yy)
                        }
                    }
                }
                maxArea = maxOf(currentComponentSize, maxArea)
            }
        }
        return maxArea
    }

    // 200. Number of Islands
    fun numIslands(grid: Array<CharArray>): Int {
        val m = grid.size
        val n = grid[0].size
        var numOfIslands = 0
        val directions = arrayOf(intArrayOf(-1, 0), intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(0, -1))
        val queue = ArrayDeque<Pair<Int, Int>>()
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (grid[i][j] != '1') continue
                numOfIslands++
                queue += Pair(i, j)
                grid[i][j] = '0'
                while(!queue.isEmpty()) {
                    val (x, y) = queue.removeFirst()
                    for ((dx, dy) in directions) {
                        val xx = x + dx
                        val yy = y + dy
                        if (xx in 0 until m && yy in 0 until n && grid[xx][yy] == '1') {
                            grid[xx][yy] = '0'
                            queue += Pair(xx, yy)
                        }
                    }
                }
            }
        }
        return numOfIslands
    }

    // 567. Permutation in String
    fun checkInclusion(s1: String, s2: String): Boolean {
        if (s1.length > s2.length) return false
        val alphabetSize = 26
        val charOccurrenceCountInSearch = IntArray(alphabetSize)
        val charOccurrenceCountInTarget = IntArray(alphabetSize)
        val offset = 'a'.toInt()
        for (i in s1.indices) {
            charOccurrenceCountInSearch[s1[i].toInt() - offset]++
            charOccurrenceCountInTarget[s2[i].toInt() - offset]++
        }

        var matchingCharGroups = 0
        for (i in 0 until alphabetSize) {
            if (charOccurrenceCountInSearch[i] == charOccurrenceCountInTarget[i]) matchingCharGroups++
        }

        for (i in 0 until s2.length - s1.length) {
            if (matchingCharGroups == alphabetSize) return true
            val left = s2[i].toInt() - offset
            val right = s2[i + s1.length].toInt() - offset

            when(++charOccurrenceCountInTarget[right] - charOccurrenceCountInSearch[right]) {
                0 -> ++matchingCharGroups
                1 -> --matchingCharGroups
            }

            when(--charOccurrenceCountInTarget[left] - charOccurrenceCountInSearch[left]) {
                0 -> ++matchingCharGroups
                -1 -> --matchingCharGroups
            }
        }
        return matchingCharGroups == alphabetSize
    }

    // 226. Invert Binary Tree
    fun invertTree(root: TreeNode?): TreeNode? = root?.apply {
        val leftPrev = left
        left = invertTree(right)
        right = invertTree(leftPrev)
    }

    // 617. Merge Two Binary Trees
    fun mergeTrees(root1: TreeNode?, root2: TreeNode?): TreeNode? = when {
        root1 == null -> if (root2 == null) null else TreeNode(root2.`val`)
        root2 == null -> TreeNode(root1.`val`)
        else -> TreeNode(root1.`val` + root2.`val`)
    }?.apply {
        left = mergeTrees(root1?.left, root2?.left)
        right = mergeTrees(root1?.right, root2?.right)
    }

    // 116. Populating Next Right Pointers in Each Node
    fun connect(root: NodeWithSibling?): NodeWithSibling? {
        if (root == null) return null
        val queue = ArrayDeque<NodeWithSibling>().apply { add(root) }
        while (!queue.isEmpty()) {
            var prev: NodeWithSibling? = null
            repeat(queue.size) { _ ->
                val node = queue.removeFirst()
                node.left?.let { queue.addLast(it) }
                node.right?.let { queue.addLast(it) }
                prev?.next = node
                prev = node
            }
        }
        return root
    }

    fun connectRecursive(root: NodeWithSibling?): NodeWithSibling? = root?.apply {
        left?.next = right
        right?.next = next?.left
        connectRecursive(right)
        connectRecursive(left)
    }

    // 509. Fibonacci Number
    fun fib(n: Int): Int {
        if (n <= 1) return n
        var a = 0
        var b = 1
        repeat(n - 2) {
            val prevB = b
            b += a
            a = prevB
        }
        return a + b
    }

    // 70. Climbing Stairs
    fun climbStairs(n: Int): Int = fib(n + 1)
}

class ListNode(var `val`: Int) {
    var next: ListNode? = null

    override fun toString(): String {
        return "val=${`val`}, next ${next}"
    }
}

class Node(var `val`: Int) {
    var children: List<Node?> = listOf()
}

class TreeNode(var `val`: Int) {
    var left: TreeNode? = null
    var right: TreeNode? = null
}

class NodeWithSibling(var `val`: Int) {
    var left: NodeWithSibling? = null
    var right: NodeWithSibling? = null
    var next: NodeWithSibling? = null
}

open class VersionControl(private val brokenVersion: Int) {
    fun isBadVersion(version: Int) : Boolean = version >= brokenVersion
}

class VCSolution(brokenVersion: Int): VersionControl(brokenVersion) {
    fun firstBadVersion(n: Int) : Int {
        var low = 1L
        var high = n.toLong()
        while (low <= high) {
            val underTest = ((low + high) / 2).toInt()
            val isBad = isBadVersion(underTest)
            when {
                !isBad -> low = underTest + 1L
                underTest > 1 && isBadVersion(underTest - 1) -> high = underTest - 1L
                else -> return underTest
            }
        }
        return -1
    }
}