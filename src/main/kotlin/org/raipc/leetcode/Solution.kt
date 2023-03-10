package org.raipc.leetcode

import java.util.*
import java.util.regex.Pattern

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

    // 2574. Left and Right Sum Differences
    fun leftRightDifference(nums: IntArray): IntArray {
        var rightSum = nums.sum()
        var leftSum = 0
        var prev = 0
        return IntArray(nums.size) {
            leftSum += prev
            prev = nums[it]
            rightSum -= prev
            Math.abs(rightSum - leftSum)
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

    // 33. Search in Rotated Sorted Array
    fun searchInRotatedArray(nums: IntArray, target: Int): Int {
        var low = 0
        var high = nums.size - 1
        while (low <= high) {
            val mid = (low + high) / 2
            when {
                nums[mid] == target -> return mid
                nums[low] <= nums[mid] -> if (target >= nums[low] && target <= nums[mid]) high = mid - 1 else low = mid + 1
                else -> if (target >= nums[mid] && target <= nums[high]) low = mid + 1 else high = mid - 1
            }
        }
        return -1
    }

    // 153. Find Minimum in Rotated Sorted Array
    fun findMin(nums: IntArray): Int {
        if (nums.size == 1) return nums[0]
        var low = 0
        var high = nums.lastIndex
        while (low < high) {
            val mid = (low + high) / 2
            if (mid > 0 && nums[mid] < nums[mid - 1]) return nums[mid]
            if (nums[low] <= nums[mid] && nums[mid] > nums[high]) low = mid + 1 else high = mid - 1
        }
        return nums[low]
    }

    // 162. Find Peak Element
    // 852. Peak Index in a Mountain Array
    fun findPeakElement(nums: IntArray): Int {
        var low = 0
        var high = nums.lastIndex
        while (low < high) {
            val mid = low + (high - low) / 2
            if (nums[mid] < nums[mid + 1]) low = mid + 1 else high = mid
        }
        return low
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

    // 34. Find First and Last Position of Element in Sorted Array
    fun searchRange(nums: IntArray, target: Int): IntArray {
        fun indexOfElement(first: Boolean): Int {
            var low = 0
            var high = nums.size - 1
            var result = -1
            while (low <= high) {
                val mid = low + (high - low) / 2
                val midVal = nums[mid]
                when {
                    midVal > target -> high = mid - 1
                    midVal < target -> low = mid + 1
                    else -> {
                        result = mid
                        if (first) high = mid - 1 else low = mid + 1
                    }
                }
            }
            return result
        }
        val first = indexOfElement(first = true)
        val last = if (first == -1) -1 else indexOfElement(first = false)
        return intArrayOf(first, last)
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

    // 12. Integer to Roman
    fun intToRoman(num: Int): String {
        val divisors = intArrayOf(1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
        val romans = arrayOf("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I")
        return buildString {
            var acc = num
            divisors.forEachIndexed { i, divisor ->
                val quotient = acc / divisor
                acc -= quotient * divisor
                repeat(quotient) {
                    append(romans[i])
                }
            }
        }
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
        var prev: ListNode? = null
        var cur = head

        while (cur != null) {
            val next = cur.next
            cur.next = prev
            prev = cur
            cur = next
        }
        return prev
    }

    // 234. Palindrome Linked List
    fun isPalindrome(head: ListNode?): Boolean {
        if (head?.next == null) return true
        var slow: ListNode = head
        var fast = head.next
        while (fast?.next != null) {
            slow = slow.next!!
            fast = fast.next!!.next
        }
        val rev: ListNode = reverseList(slow.next)!!
        slow = head
        fast = rev
        while (fast != null) {
            if (slow.`val` != fast.`val`) {
                return false
            }
            slow = slow.next!!
            fast = fast.next
        }
        return true
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
    fun twoSum2(numbers: IntArray, target: Int): IntArray {
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

    // 15. 3Sum
    fun threeSum(nums: IntArray) = mutableListOf<List<Int>>().apply {
        nums.sort()
        for (i in 0 until nums.size - 2) {
            if (i == 0 || i > 0 && nums[i] != nums[i - 1]) {
                var low = i + 1
                var high: Int = nums.size - 1
                val sum = -nums[i]
                while (low < high) {
                    if (nums[low] + nums[high] == sum) {
                        add(listOf(nums[i], nums[low], nums[high]))
                        while (low < high && nums[low] == nums[low + 1]) low++
                        while (low < high && nums[high] == nums[high - 1]) high--
                        low++
                        high--
                    } else if (nums[low] + nums[high] < sum) low++ else high--
                }
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

    // 144. Binary Tree Preorder Traversal
    fun preorderTraversal(root: TreeNode?): List<Int> {
        if (root == null) return emptyList()
        val stack = ArrayDeque<TreeNode>().apply { add(root) }
        return mutableListOf<Int>().apply {
            while (stack.isNotEmpty()) {
                val node = stack.pollLast()!!
                add(node.`val`)
                node.right?.let { stack.add(it) }
                node.left?.let { stack.add(it) }
            }
        }
    }

    // 145. Binary Tree Postorder Traversal
    fun postorderTraversal(root: TreeNode?): List<Int> =
        if (root == null) emptyList() else ArrayList<Int>().apply {
            val stack = ArrayDeque<TreeNode>().apply { add(root) }
            var curr = root
            while (stack.isNotEmpty()) {
                while (curr != null) {
                    stack.addLast(curr)
                    add(curr.`val`)
                    curr = curr.right
                }
                curr = stack.removeLast().left
            }
        }.asReversed()

    // 94. Binary Tree Inorder Traversal
    fun inorderTraversal(root: TreeNode?): List<Int> =
        if (root == null) emptyList() else mutableListOf<Int>().apply {
            val stack = ArrayDeque<TreeNode>()
            var cur = root
            while (cur != null || stack.isNotEmpty()) {
                while (cur != null) {
                    stack.addLast(cur)
                    cur = cur.left
                }
                cur = stack.removeLast()
                add(cur.`val`)
                cur = cur.right
            }
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

    // 24. Swap Nodes in Pairs
    fun swapPairs(head: ListNode?) = reverseKGroup(head, 2)

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
                node.left.let { it == null || it.`val` < value && isValidRecursive(it, rangeLeft, value.toLong()) } &&
                node.right.let { it == null || it.`val` > value && isValidRecursive(it, value.toLong(), rangeRight) }
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

    // 438. Find All Anagrams in a String
    fun findAnagrams(s: String, p: String): List<Int> {
        if (p.length > s.length) return emptyList()
        val alphabetSize = 26
        val charOccurrenceCountInSearch = IntArray(alphabetSize)
        val charOccurrenceCountInTarget = IntArray(alphabetSize)
        val offset = 'a'.toInt()
        for (i in p.indices) {
            charOccurrenceCountInSearch[p[i].toInt() - offset]++
            charOccurrenceCountInTarget[s[i].toInt() - offset]++
        }

        var matchingCharGroups = 0
        for (i in 0 until alphabetSize) {
            if (charOccurrenceCountInSearch[i] == charOccurrenceCountInTarget[i]) matchingCharGroups++
        }

        val result = mutableListOf<Int>().also { if (matchingCharGroups == alphabetSize) it.add(0) }

        for (i in 0 until s.length - p.length) {
            val left = s[i].toInt() - offset
            val right = s[i + p.length].toInt() - offset

            when(++charOccurrenceCountInTarget[right] - charOccurrenceCountInSearch[right]) {
                0 -> ++matchingCharGroups
                1 -> --matchingCharGroups
            }

            when(--charOccurrenceCountInTarget[left] - charOccurrenceCountInSearch[left]) {
                0 -> ++matchingCharGroups
                -1 -> --matchingCharGroups
            }
            if (matchingCharGroups == alphabetSize) result.add(i + 1)
        }
        return result
    }

    // 30. Substring with Concatenation of All Words
    fun findSubstring(s: String, words: Array<String>): List<Int> {
        if (words.isEmpty() || s.length < words.size * words[0].length) return emptyList()
        val step = words[0].length
        val permutationLength = step * words.size
        val result = mutableListOf<Int>()
        val wordsMap = hashMapOf<String, Int>()
        words.forEach { wordsMap.compute(it) { _, prev -> (prev ?: 0) + 1 } }

        for (offset in 0 until step) {
            if (s.length < permutationLength + offset) break
            val matches = hashMapOf<String, Int>()
            for (i in offset until permutationLength + offset step step) {
                matches.compute(s.substring(i, i + step)) { _, prev -> (prev ?: 0) + 1 }
            }
            var numberOfMatchingWords = 0
            matches.forEach { (word, count) ->
                numberOfMatchingWords += minOf(wordsMap.getOrDefault(word, 0), count)
            }
            if (numberOfMatchingWords == words.size) result.add(offset)

            for (i in offset + step .. s.length - permutationLength step step) {
                val toRemove = s.substring(i-step, i)
                val toAdd = s.substring(i+permutationLength-step, i+permutationLength)
                if (toRemove != toAdd) {
                    val numOfRemoveInWords = wordsMap[toRemove] ?: 0
                    if (numOfRemoveInWords > 0 &&
                        numOfRemoveInWords > ((matches.compute(toRemove) { _, prev -> if (prev == null) null else prev - 1 }) ?: 0)) {
                        --numberOfMatchingWords
                    }
                    val numOfAddInWords = wordsMap[toAdd] ?: 0
                    if (numOfAddInWords > 0) {
                        val newCount = matches.compute(toAdd) { _, prev -> if (prev == null) 1 else prev + 1 } ?: 0
                        if (newCount <= numOfAddInWords) ++numberOfMatchingWords
                    }
                }
                if (numberOfMatchingWords == words.size) result.add(i)
            }
        }
        return result
    }

    // 226. Invert Binary Tree
    fun invertTree(root: TreeNode?): TreeNode? = root?.apply {
        val leftPrev = left
        left = invertTree(right)
        right = invertTree(leftPrev)
    }

    // 110. Balanced Binary Tree
    fun isBalanced(root: TreeNode?): Boolean {
        fun height(node: TreeNode?): Int {
            if (node == null) return 0
            val leftHeight = height(node.left)
            val rightHeight = height(node.right)
            return when {
                leftHeight == -1 -> -1
                rightHeight == -1 -> -1
                Math.abs(leftHeight - rightHeight) > 1 -> -1
                else -> Math.max(leftHeight, rightHeight) + 1
            }
        }
        return root == null || height(root) != -1
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

    // 542. 01 Matrix
    fun updateMatrix(mat: Array<IntArray>): Array<IntArray> {
        val n = mat.size
        val m = mat[0].size
        val result = Array(n) { IntArray(m) { Int.MAX_VALUE} }
        val directions = arrayOf(intArrayOf(-1, 0), intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(0, -1))
        val queue = ArrayDeque<Pair<Int, Int>>()
        for (i in 0 until n) {
            for (j in 0 until m) {
                if (mat[i][j] == 0) {
                    result[i][j] = 0
                    queue += Pair(i, j)
                }
            }
        }
        while (!queue.isEmpty()) {
            val (x, y) = queue.removeFirst()
            for ((dx, dy) in directions) {
                val i = x + dx
                val j = y + dy
                if (i in 0 until n && j in 0 until m && result[i][j] > result[x][y] + 1) {
                    result[i][j] = result[x][y] + 1
                    queue += Pair(i, j)
                }
            }
        }
        return result
    }

    // 103. Binary Tree Zigzag Level Order Traversal
    fun zigzagLevelOrder(root: TreeNode?): List<List<Int>> {
        if (root == null) return emptyList()
        val deque = ArrayDeque<TreeNode>().apply { add(root) }
        val result = mutableListOf<MutableList<Int>>()
        while (!deque.isEmpty()) {
            val list = ArrayList<Int>(deque.size).also { result.add(it) }
            repeat(deque.size) {
                val node = deque.removeFirst()
                list += node.`val`
                node.left?.let { deque += it }
                node.right?.let { deque += it }
            }
        }
        for (i in 1 until result.size step 2) {
            result[i] = result[i].asReversed()
        }
        return result
    }

    // 994. Rotting Oranges
    fun orangesRotting(grid: Array<IntArray>): Int {
        val queue = ArrayDeque<Pair<Int, Int>>()
        val n = grid.size
        val m = grid[0].size
        for (i in 0 until n) {
            for (j in 0 until m) {
                if (grid[i][j] == 2) queue += Pair(i, j)
            }
        }
        val directions = arrayOf(intArrayOf(-1, 0), intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(0, -1))
        var minutes = -1
        while (queue.isNotEmpty()) {
            minutes++
            repeat(queue.size) {
                val (x, y) = queue.removeFirst()
                for ((dx, dy) in directions) {
                    val i = x + dx
                    val j = y + dy
                    if (i in 0 until n && j in 0 until m && grid[i][j] == 1) {
                        grid[i][j] = 2
                        queue.add(Pair(i, j))
                    }
                }
            }
        }
        return if (grid.any { it.contains(1) }) -1 else maxOf(minutes, 0)
    }

    // 746. Min Cost Climbing Stairs
    fun minCostClimbingStairs(cost: IntArray): Int {
        for (i in 2 until cost.size) {
            cost[i] += minOf(cost[i-2], cost[i-1])
        }
        return minOf(cost[cost.lastIndex], cost[cost.lastIndex - 1])
    }

    // 62. Unique Paths
    fun uniquePaths(m: Int, n: Int): Int {
        val results = Array(m) { IntArray(n) }
        for (i in 1 until m) { results[i][0] = 1 }
        for (j in 1 until n) { results[0][j] = 1 }
        for (i in 1 until m) {
            for (j in 1 until n) {
                results[i][j] = results[i][j-1] + results[i-1][j]
            }
        }
        return results.last().last()
    }

    fun uniquePathsMath(m: Int, n: Int): Int {
        var result = 1L
        var left = 1
        var right = m + n - 2
        val max = maxOf(m, n)
        while (right >= max) {
            result = result * right / left
            right--
            left++
        }
        return result.toInt()
    }

    // 11. Container With Most Water
    fun maxArea(height: IntArray): Int {
        var result = 0
        var left = 0
        var right = height.lastIndex
        while (left < right) {
            result = maxOf(result, minOf(height[left], height[right]) * (right - left))
            if (height[left] <= height[right]) {
                left++
            } else {
                right--
            }
        }
        return result
    }

    // 2517. Maximum Tastiness of Candy Basket
    fun maximumTastiness(price: IntArray, k: Int): Int {
        price.sort()
        var left = 0
        var right = price.last() - price.first() + 1
        fun maxSize(mid: Int): Int {
            var size = 0
            var prevPrice = -mid
            for (p in price) if (p >= prevPrice + mid) {
                prevPrice = p
                ++size
            }
            return size
        }
        while (left < right) {
            val mid = (left + right) / 2;
            if (maxSize(mid) >= k) left = mid + 1 else right = mid
        }
        return left - 1
    }

    // 794. Valid Tic-Tac-Toe State
    fun validTicTacToe(board: Array<String>): Boolean {
        val xCounts = board.sumBy { row -> row.count { it == 'X' } }
        val oCounts = board.sumBy { row -> row.count { it == 'O' } }
        val numberOfWinsInRows = (0 until 3).count { board[it][0] == board[it][1] && board[it][0] == board[it][2] && board[it][0] != ' ' }
        val numberOfWinsInCols = (0 until 3).count { board[0][it] == board[1][it] && board[0][it] == board[2][it] && board[0][it] != ' '}
        if (numberOfWinsInRows > 1 || numberOfWinsInCols > 1) return false
        val center = board[1][1]
        val diagonalWin = center != ' ' && ((board[0][0] == center && board[2][2] == center) ||
                (board[0][2] == center && board[2][0] == center))
        return when {
            diagonalWin -> if (center == 'X') xCounts == oCounts + 1 else xCounts == oCounts
            numberOfWinsInRows > 0 -> if (board.contains("XXX")) xCounts == oCounts + 1 else xCounts == oCounts
            numberOfWinsInCols > 0 -> if ((0..2).any { board[0][it] == 'X' && board[1][it] == 'X' && board[2][it] == 'X'})
                xCounts == oCounts + 1 else xCounts == oCounts
            else -> xCounts == oCounts || xCounts == oCounts + 1
        }
    }

    // 540. Single Element in a Sorted Array
    fun singleNonDuplicate(nums: IntArray): Int {
        val size = nums.size
        if (size == 1) return nums.first()
        var left = 0
        var right = nums.size
        while (left < right) {
            val mid = ((right + left) / 2).let { if (it % 2 == 1) it - 1 else it }
            val value = nums[mid]
            if (mid + 1 < size && value == nums[mid+1]) {
                left = mid + 2
            } else if (mid == 0 || value != nums[mid-1]) {
                return value
            } else {
                right = mid
            }
        }
        return nums.last()
    }

    // 424. Longest Repeating Character Replacement
    fun characterReplacement(s: String, k: Int): Int {
        val charCountMap = IntArray(26)
        val offset = 'A'.toInt()
        var maxCount = 0
        var left = 0
        for (right in s.indices) {
            val idx = s[right].toInt() - offset
            charCountMap[idx]++
            maxCount = maxOf(maxCount, charCountMap[idx])
            if (right - left + 1 - k > maxCount) {
                charCountMap[s[left].toInt() - offset]--
                ++left
            }
        }
        return s.lastIndex - left + 1
    }

    // 77. Combinations
    fun combine(n: Int, k: Int): List<List<Int>> {
        fun doCombine(position: Int, acc: Array<Int>, result: MutableList<List<Int>>) {
            val prev = if (position == 0) 0 else acc[position-1]
            for (i in prev+1..n) {
                acc[position] = i
                if (position + 1 == k) {
                    result.add(ArrayList<Int>(k).apply { addAll(acc) })
                } else {
                    doCombine(position + 1, acc, result)
                }
            }
        }
        return mutableListOf<List<Int>>().apply { doCombine(0, Array(k){ 0 }, this) }
    }

    // 46. Permutations
    fun permute(nums: IntArray): List<List<Int>> {
        fun doPermute(index: Int, acc: MutableList<Int>, result: MutableList<List<Int>>) {
            if (index == acc.lastIndex) result.add(ArrayList<Int>().apply { addAll(acc) })
            else for (i in index until acc.size) {
                if (i == index) {
                    doPermute(index + 1, acc, result)
                } else {
                    Collections.swap(acc, index, i)
                    doPermute(index + 1, acc, result)
                    Collections.swap(acc, index, i)
                }
            }
        }
        return mutableListOf<List<Int>>().apply { doPermute(0, nums.toMutableList(), this) }
    }

    // 784. Letter Case Permutation
    fun letterCasePermutation(s: String): List<String> {
        fun doPermute(arr: CharArray, index: Int, result: MutableList<String>) {
            if (index == arr.size) {
                result.add(String(arr))
            } else {
                val ch = arr[index]
                if (ch in '0'..'9') {
                    doPermute(arr, index + 1, result)
                } else {
                    arr[index] = ch.toUpperCase()
                    doPermute(arr, index + 1, result)
                    arr[index] = ch.toLowerCase()
                    doPermute(arr, index + 1, result)
                }
            }
        }
        return mutableListOf<String>().apply { doPermute(s.toCharArray(), 0, this) }
    }

    // 41. First Missing Positive
    fun firstMissingPositive(nums: IntArray): Int {
        val n = nums.size
        for (i in 0 until n) {
            while (nums[i] in 1..n && nums[i] != nums[nums[i] - 1]) {
                val temp = nums[nums[i] - 1]
                nums[nums[i] - 1] = nums[i]
                nums[i] = temp
            }
        }
        for (i in 0 until n) {
            if (nums[i] != i + 1) return i + 1
        }
        return n + 1
    }

    // 136. Single Number
    fun singleNumber(nums: IntArray): Int {
        var acc = 0
        nums.forEach { acc = acc xor it }
        return acc
    }

    // 1011. Capacity To Ship Packages Within D Days
    fun shipWithinDays(weights: IntArray, days: Int): Int {
        fun canShipWithCapacity(capacity: Int): Boolean {
            var daysRequired = 1
            var currentWeight = 0
            for (weight in weights) {
                if (currentWeight + weight <= capacity) {
                    currentWeight += weight
                } else {
                    currentWeight = weight
                    daysRequired++
                    if (daysRequired > days) break
                }
            }
            return daysRequired <= days
        }
        var l = weights.max()!!
        var r = weights.sum()
        while (l < r) {
            val mid = (r + l) / 2
            if (canShipWithCapacity(mid)) {
                r = mid
            } else {
                l = mid + 1
            }
        }
        return l
    }

    // 1. Two Sum
    fun twoSum(nums: IntArray, target: Int): IntArray {
        val visited = hashMapOf<Int, Int>()
        nums.forEachIndexed { i, num ->
            val toAdd = target - num
            val indexOfNumberToAdd = visited[toAdd]
            if (indexOfNumberToAdd != null) {
                return intArrayOf(indexOfNumberToAdd, i)
            }
            visited[num] = i
        }
        throw IllegalStateException("Answer must exist!")
    }

    // 653. Two Sum IV - Input is a BST
    fun findTarget(root: TreeNode?, k: Int): Boolean {
        if (root == null) return false
        val set = hashSetOf<Int>()
        val queue = ArrayDeque<TreeNode>().apply { add(root) }
        while (queue.isNotEmpty()) {
            val node = queue.pollFirst()!!
            if (set.contains(k - node.`val`)) return true
            set.add(node.`val`)
            node.left?.let { queue.add(it) }
            node.right?.let { queue.add(it) }
        }
        return false
    }

    // 299. Bulls and Cows
    fun getHint(secret: String, guess: String): String {
        var bulls = 0
        var cows = 0
        val digitsInSecret = IntArray(10)
        val digitsInGuess = IntArray(10)
        secret.forEachIndexed { i, secretCh ->
            val guessCh = guess[i]
            if (secretCh == guessCh) ++bulls
            else {
                digitsInGuess[guessCh - '0']++
                digitsInSecret[secretCh - '0']++
            }
        }
        for (i in digitsInSecret.indices) { cows += minOf(digitsInSecret[i], digitsInGuess[i]) }
        return "${bulls}A${cows}B"
    }

    // 198. House Robber
    fun rob(nums: IntArray): Int {
        if (nums.size < 3) return nums.max()!!
        nums[2] += nums[0]
        for (i in 3 until nums.size) {
            nums[i] += maxOf(nums[i-2], nums[i-3])
        }
        return maxOf(nums[nums.lastIndex], nums[nums.lastIndex-1])
    }

    // 120. Triangle
    fun minimumTotal(triangle: List<List<Int>>): Int {
        val maxWidth = triangle[triangle.lastIndex].size
        if (maxWidth == 1) return triangle[0][0]
        var minsCurr = IntArray(maxWidth - 1)
        var minsPrev = IntArray(maxWidth) { triangle.last()[it] }
        for (i in triangle.lastIndex - 1 downTo 0) {
            val row = triangle[i]
            for (j in row.indices) {
                minsCurr[j] = row[j] + minOf(minsPrev[j], minsPrev[j+1])
            }
            val tmp = minsPrev
            minsPrev = minsCurr
            minsCurr = tmp
        }
        return minsPrev[0]
    }

    // 502. IPO
    fun findMaximizedCapital(k: Int, w: Int, profits: IntArray, capital: IntArray): Int {
        data class Project(val profit: Int, val capital: Int)

        val size = profits.size
        val projects = Array(size) { Project(profits[it], capital[it]) }
            .apply { sortBy { it.capital } }
        var earnings = w
        val pq = PriorityQueue<Project>(size, compareByDescending { it.profit })
        var ptr = 0
        repeat(k) {
            while (ptr < size && projects[ptr].capital <= earnings) {
                pq.add(projects[ptr])
                ++ptr
            }
            if (pq.isEmpty()) {
                return earnings
            }
            earnings += pq.poll().profit
        }
        return earnings
    }

    // 844. Backspace String Compare
    fun backspaceCompare(s: String, t: String): Boolean {
        fun moveToCaret(startPos: Int, str: String): Int {
            var idx = startPos
            var backspaces = 0
            while (backspaces >= 0 && idx >= 0) {
                if (str[idx] == '#') {
                    ++backspaces
                } else {
                    --backspaces
                }
                --idx
            }
            return if (backspaces < 0) idx + 1 else idx
        }
        var sIdx = s.length
        var tIdx = t.length
        do {
            sIdx = moveToCaret(--sIdx, s)
            tIdx = moveToCaret(--tIdx, t)
            if (sIdx >= 0 && tIdx >= 0 && s[sIdx] != t[tIdx]) {
                break
            }

        } while (sIdx >= 0 && tIdx >= 0)
        return sIdx < 0 && tIdx < 0
    }


    // 394. Decode String
    fun decodeString(s: String): String = buildString {
        val repeatStack = ArrayDeque<IntArray>()
        var numOfRepeats = 0
        for (ch in s) {
            when (ch) {
                in '0'..'9' -> numOfRepeats = numOfRepeats * 10 + (ch - '0')
                '[' -> {
                    repeatStack += intArrayOf(numOfRepeats, length)
                    numOfRepeats = 0
                }
                ']' -> {
                    val (repeats, startIdx) = repeatStack.removeLast()
                    val endIdx = length
                    repeat(repeats - 1) {
                        append(this, startIdx, endIdx)
                    }
                }
                else -> append(ch)
            }
        }
    }

    // 231. Power of Two
    fun isPowerOfTwo(n: Int): Boolean = (n and (n - 1)) == 0 && n > 0

    // 191. Number of 1 Bits
    fun hammingWeight(n:Int):Int = Integer.bitCount(n)

    // 190. Reverse Bits
    fun reverseBits(n:Int):Int = Integer.reverse(n)

    // 1675. Minimize Deviation in Array
    fun minimumDeviation(nums: IntArray): Int {
        val set = TreeSet<Int>().apply { nums.forEach { add(if (it % 2 == 1) it * 2 else it) } }
        var deviation = Integer.MAX_VALUE
        var foundMin = false
        while(!foundMin) {
            if (set.size == 1) {
                deviation = 0
                foundMin = true
            } else {
                val max = set.pollLast()!!
                deviation = minOf(deviation, max - set.first())
                if (max % 2 == 0) {
                    set.add(max / 2)
                } else {
                    foundMin = true
                }
            }
        }
        return deviation
    }

    // 1046. Last Stone Weight
    fun lastStoneWeight(stones: IntArray): Int {
        val pq = PriorityQueue<Int>(compareByDescending { it }).apply { stones.forEach { add(it) } }
        while (pq.size >= 2) {
            val diff = pq.poll() - pq.poll()
            if (diff > 0) pq.add(diff)
        }
        return if (pq.isEmpty()) 0 else pq.element()
    }

    // 692. Top K Frequent Words
    fun topKFrequent(words: Array<String>, k: Int): List<String> {
        val frequencyMap = hashMapOf<String, Int>()
        words.forEach { frequencyMap.compute(it) { _, prev -> (prev?:0) + 1 } }
        val pq = PriorityQueue<Map.Entry<String, Int>>(k + 1,
            compareBy<Map.Entry<String, Int>> { it.value }.then(compareByDescending { it.key }))
        frequencyMap.entries.forEach {
            pq += it
            if (pq.size > k) pq.poll()
        }
        return (0 until pq.size).map { pq.poll().key }.asReversed()
    }

    // 217. Contains Duplicate
    fun containsDuplicate(nums: IntArray) = hashSetOf<Int>().apply { nums.forEach { add(it) } }.size != nums.size

    // 53. Maximum Subarray
    fun maxSubArray(nums: IntArray): Int {
        var maxSum = Integer.MIN_VALUE
        var windowSum = 0
        for (num in nums) {
            windowSum += num
            if (windowSum > maxSum) {
                maxSum = windowSum
            }
            if (windowSum < 0) {
                windowSum = 0
            }
        }
        return maxSum
    }

    // 88. Merge Sorted Array
    fun merge(nums1: IntArray, m: Int, nums2: IntArray, n: Int): Unit {
        var idx1 = m - 1
        var idx2 = n - 1
        var insertionIdx = m + n - 1
        while (idx1 >= 0 && idx2 >= 0) {
            val first = nums1[idx1]
            val second = nums2[idx2]
            if (first > second) {
                nums1[insertionIdx] = first
                --idx1
            } else {
                nums1[insertionIdx] = second
                --idx2
            }
            --insertionIdx
        }
        while (idx2 >= 0) {
            nums1[insertionIdx] = nums2[idx2]
            --insertionIdx
            --idx2
        }
    }

    // 72. Edit Distance
    fun minDistance(word1: String, word2: String): Int {
        val len1 = word1.length
        val len2 = word2.length
        return when {
            len1 == 0 -> len2
            len2 == 0 -> len1
            len1 > len2 -> minDistance(word2, word1)
            else -> {
                val dp = IntArray(len1 + 1) { it }
                for (j in 1..len2) {
                    var prev = dp[0]
                    dp[0] += 1
                    val ch2 = word2[j-1]
                    for (i in 1..len1) {
                        val temp = dp[i]
                        dp[i] = if (word1[i-1] == ch2) prev else minOf(prev + 1, minOf(dp[i - 1] + 1, dp[i] + 1))
                        prev = temp
                    }
                }
                dp.last()
            }
        }
    }

    // 350. Intersection of Two Arrays II
    fun intersect(nums1: IntArray, nums2: IntArray): IntArray {
        val smallArraySize = nums2.size
        if (nums1.size < smallArraySize) return intersect(nums2, nums1)
        val counterMap = IntArray(1001)
        nums2.forEach { ++counterMap[it] }
        val result = IntArray(smallArraySize)
        var resultSize = 0
        for (num in nums1) {
            val cntLeft = --counterMap[num]
            if (cntLeft >= 0) {
                result[resultSize++] = num
                if (resultSize == smallArraySize) break
            }
        }
        return if (resultSize == smallArraySize) result else result.copyOfRange(0, resultSize)
    }

    // 427. Construct Quad Tree
    fun construct(grid: Array<IntArray>): QuadTreeNode? {
        val leafZero = QuadTreeNode(`val` = false, isLeaf = true)
        val leafOne = QuadTreeNode(`val` = true, isLeaf = true)
        val nodes = Array(3) { leafZero }
        fun construct(topIdx: Int, leftIdx: Int, length: Int): QuadTreeNode {
            if (length == 1) return if (grid[topIdx][leftIdx] == 0) leafZero else leafOne
            val halfLength = length / 2
            val topLeft = construct(topIdx, leftIdx, halfLength)
            val topRight = construct(topIdx, leftIdx + halfLength, halfLength)
            val bottomLeft = construct(topIdx + halfLength, leftIdx, halfLength)
            val bottomRight = construct(topIdx + halfLength, leftIdx + halfLength, halfLength)
            nodes[0] = topRight
            nodes[1] = bottomLeft
            nodes[2] = bottomRight
            return if (topLeft.isLeaf && nodes.all { it.isLeaf && it.`val` == topLeft.`val` }) topLeft
                else QuadTreeNode(`val` = true, isLeaf = false).apply {
                    this.topLeft = topLeft
                    this.topRight = topRight
                    this.bottomLeft = bottomLeft
                    this.bottomRight = bottomRight
            }
        }
        return construct(0, 0, grid.size)
    }

    // 118. Pascal's Triangle
    fun generate(numRows: Int): List<List<Int>> {
        if (numRows == 1) return listOf(listOf(1))
        if (numRows == 2) return listOf(listOf(1), listOf(1, 1))
        val result = Array(numRows) { Array(it + 1) { 1 }.asList() as MutableList<Int> }
        for (cnt in 2 until numRows) {
            result[cnt].apply {
                val prev = result[cnt-1]
                for (i in 1 until cnt) this[i] = prev[i-1] + prev[i]
            }
        }
        return result.asList()
    }

    // 566. Reshape the Matrix
    fun matrixReshape(mat: Array<IntArray>, r: Int, c: Int): Array<IntArray> {
        val n = mat.size
        val m = mat[0].size
        if (n * m != r * c || n == r) return mat
        return Array(r) { IntArray(c) }.apply {
            for (i in 0 until r * c) this[i / c][i % c] = mat[i / m][i % m]
        }
    }

    // 387. First Unique Character in a String
    fun firstUniqChar(s: String): Int {
        val countMap = IntArray(26)
        for (ch in s) { ++countMap[ch-'a'] }
        return s.indexOfFirst { countMap[it-'a'] == 1 }
    }

    // 383. Ransom Note
    fun canConstruct(ransomNote: String, magazine: String): Boolean {
        val magazineChars = IntArray(26)
        magazine.forEach { ++magazineChars[it-'a'] }
        ransomNote.forEach { if (--magazineChars[it-'a'] < 0) return false }
        return true
    }

    // 242. Valid Anagram
    fun isAnagram(s: String, t: String): Boolean {
        if (s.length != t.length) return false
        val charCount = IntArray(26)
        s.forEach { ++charCount[it-'a'] }
        t.forEach { --charCount[it-'a'] }
        return charCount.all { it == 0 }
    }

    // 27. Remove Element
    fun removeElement(nums: IntArray, `val`: Int): Int {
        var shiftCnt = 0
        nums.forEachIndexed { i, num -> if (num == `val`) ++shiftCnt else if (shiftCnt > 0) nums[i-shiftCnt] = num }
        return nums.size - shiftCnt
    }

    // 652. Find Duplicate Subtrees
    fun findDuplicateSubtrees(root: TreeNode?): List<TreeNode?> {
        fun traverse(node: TreeNode?, result: MutableList<TreeNode>, counter: HashMap<String, Int>): String {
            if (node == null) return ""
            val left = traverse(node.left, result, counter)
            val right = traverse(node.right, result, counter)
            return ("" + node.`val` + "L" + left + "R" + right)
                .also { if (counter.compute(it) { _, prev -> (prev?:0) + 1 } == 2) result += node }
        }
        return mutableListOf<TreeNode>().apply { traverse(root, this, hashMapOf()) }
    }

    // 74. Search a 2D Matrix
    fun searchMatrix(matrix: Array<IntArray>, target: Int): Boolean {
        val n = matrix[0].size
        var left = 0
        var right = matrix.size * n - 1
        do {
            val mid = (right + left) / 2
            val value = matrix[mid / n][mid % n]
            when {
                value < target -> left = mid + 1
                value > target -> right = mid - 1
                else -> return true
            }
        } while (left <= right)
        return false
    }

    // 36. Valid Sudoku
    fun isValidSudoku(board: Array<CharArray>): Boolean {
        for (i in 0 until 9) {
            val setByRow = IntArray(10)
            val setByCol = IntArray(10)
            for (j in 0 until 9) {
                board[j][i].let { if (it != '.' && ++setByRow[it-'0'] > 1) return false }
                board[i][j].let { if (it != '.' && ++setByCol[it-'0'] > 1) return false }
            }
        }
        for (row in 0 until 9 step 3) {
            for (col in 0 until 9 step 3) {
                val set = IntArray(10)
                for (i in 0 until 3) {
                    for (j in 0 until 3) {
                        board[row + i][col + j].let { if (it != '.' && ++set[it-'0'] > 1) return false }
                    }
                }
            }
        }
        return true
    }

    // 17. Letter Combinations of a Phone Number
    fun letterCombinations(digits: String): List<String> {
        if (digits.isEmpty()) return emptyList()
        val keyboard = arrayOf("", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz")
        fun combine(index: Int, buf: CharArray, result: MutableList<String>) {
            if (index == digits.length) result += String(buf)
            else for (ch in keyboard[digits[index]-'0']) combine(index + 1, buf.apply { this[index] = ch }, result)
        }
        return mutableListOf<String>().apply { combine(0, CharArray(digits.length), this) }
    }

    // 58. Length of Last Word
    fun lengthOfLastWord(s: String) = s.trim().let { it.length - it.lastIndexOf(' ') - 1 }

    // 28. Find the Index of the First Occurrence in a String
    fun strStr(haystack: String, needle: String) = with(Pattern.compile(needle, Pattern.LITERAL).matcher(haystack))
        { if (find()) start() else -1 }

    private fun swap(nums: IntArray, firstIdx: Int, secondIdx: Int) {
        val tmp = nums[firstIdx]
        nums[firstIdx] = nums[secondIdx]
        nums[secondIdx] = tmp
    }

    // 912. Sort an Array
    fun sortArray(nums: IntArray): IntArray {
        fun merge(begin: Int, mid: Int, end: Int) {
            var left = begin
            var leftEnd = mid - 1
            var right = mid
            if (nums[leftEnd] > nums[right]) {
                while (left <= leftEnd && right < end) {
                    if (nums[left] > nums[right]) {
                        val value = nums[right]
                        for (i in right downTo left + 1) nums[i] = nums[i - 1]
                        nums[left] = value
                        leftEnd++
                        right++
                    }
                    left++
                }
            }
        }
        fun sortSubArray(begin: Int, end: Int) {
            if (end - begin <= 8) {
                for (i in begin until end - 1) {
                    var minIdx = i
                    for (j in i + 1 until end) {
                        if (nums[j] < nums[minIdx]) minIdx = j
                    }
                    swap(nums, i, minIdx)
                }
            } else {
                val mid = (begin + end) / 2
                sortSubArray(begin, mid)
                sortSubArray(mid, end)
                merge(begin, mid, end)
            }
        }
        return nums.apply { sortSubArray(0, this.size) }
    }

    // 443. String Compression
    fun compress(chars: CharArray): Int {
        fun putCharCompressed(ch: Char, cnt: Int, index: Int): Int = index + 1 + when {
            cnt == 1 -> 0
            cnt < 10 -> 1
            cnt < 100 -> 2
            cnt < 1000 -> 3
            else -> 4
        }.also {
            chars[index] = ch
            var t = cnt
            for (i in it downTo 1) {
                chars[index + i] = '0' + (t % 10)
                t /= 10
            }
        }
        var updatedIndex = 0
        var groupCount = 0
        var groupChar = chars[0]
        chars.forEach { ch ->
            if (ch == groupChar) ++groupCount
            else {
                updatedIndex = putCharCompressed(groupChar, groupCount, updatedIndex)
                groupChar = ch
                groupCount = 1
            }
        }
        return putCharCompressed(groupChar, groupCount, updatedIndex)
    }

    // 141. Linked List Cycle
    fun hasCycle(head: ListNode?): Boolean {
        if (head == null) return false
        var fast = head
        var slow = head
        do {
            fast = fast!!.next?.next
            slow = slow!!.next
        } while (fast != null && fast != slow)
        return fast != null
    }

    // 203. Remove Linked List Elements
    fun removeElements(head: ListNode?, `val`: Int): ListNode? {
        val tmp = ListNode(0)
        tmp.next = head
        var current = tmp
        var next = tmp.next
        while (next != null) {
            if (next.`val` == `val`) current.next = next.next
            else current = next
            next = current.next
        }
        return tmp.next
    }

    // 83. Remove Duplicates from Sorted List
    fun deleteDuplicates(head: ListNode?): ListNode? {
        if (head != null) {
            var current: ListNode = head
            var next = current.next
            while (next != null) {
                if (next.`val` == current.`val`) current.next = next.next
                else current = next
                next = current.next
            }
        }
        return head
    }

    // 82. Remove Duplicates from Sorted List II
    fun deleteDuplicates2(head: ListNode?): ListNode? {
        if (head == null) return null
        val tempHead = ListNode(0).apply { next = head }
        var previous: ListNode = tempHead
        var current: ListNode = head
        var next = current.next
        var toDelete = Integer.MAX_VALUE
        while (next != null) {
            if (next!!.`val` == current.`val`) {
                previous.next = next!!.next
                toDelete = current.`val`
            } else if (current.`val` != toDelete) {
                previous = current
            }
            current = next!!
            next = next!!.next
        }
        return tempHead.next
    }

    // 202. Happy Number
    fun isHappy(n: Int): Boolean {
        fun sumOfSquares(value: Int): Int {
            var acc = 0
            var t = value
            while (t > 0) {
                acc += (t % 10).let { it * it }
                t /= 10
            }
            return acc
        }
        var slow = n
        var fast = n
        do {
            fast = sumOfSquares(sumOfSquares(fast))
            slow = sumOfSquares(slow)
        } while (slow != fast)
        return fast == 1
    }

    // 1706. Where Will the Ball Fall
    fun findBall(grid: Array<IntArray>): IntArray {
        val resultSize = grid[0].size
        val lastIdx = resultSize - 1
        val output = IntArray(resultSize) { it }
        for (cell in grid) {
            output.forEachIndexed { index, pos ->
                output[index] = when {
                    pos == -1 -> -1
                    pos == 0 && cell[pos] == -1 || pos == lastIdx && cell[pos] == 1 -> -1
                    cell[pos] == 1 && cell[pos] + cell[pos+1] != 0 -> pos + 1
                    cell[pos] == -1 && cell[pos] + cell[pos-1] != 0 -> pos - 1
                    else -> -1
                }
            }
        }
        return output
    }

    // 54. Spiral Matrix
    fun spiralOrder(matrix: Array<IntArray>): List<Int> {
        var firstRow = 0
        var lastRow = matrix.lastIndex
        var firstCol = 0
        var lastCol = matrix[0].lastIndex
        return ArrayList<Int>(matrix.size * matrix[0].size).apply {
            while (firstRow <= lastRow && firstCol <= lastCol) {
                matrix[firstRow].let { for (i in firstCol..lastCol) add(it[i]) }.also { firstRow++ }
                for (j in firstRow..lastRow) { add(matrix[j][lastCol]) }; lastCol--
                if (firstRow <= lastRow) {
                    if (lastCol < 0) break
                    matrix[lastRow].let { for (i in lastCol downTo firstCol) add(it[i]) }.also { lastRow-- }
                    if (firstCol <= lastCol) {
                        for (j in lastRow downTo firstRow) { add(matrix[j][firstCol]) }; firstCol++
                    }
                }
            }
        }
    }

    // 43. Multiply Strings
    fun multiply(num1: String, num2: String): String {
        if (num1.length + num2.length < 19) return "" + (num1.toLong() * num2.toLong())
        val resultDigits = IntArray(num1.length + num2.length)
        for (i in num1.lastIndex downTo 0) {
            val digit1 = num1[i] - '0'
            for (j in num2.lastIndex downTo 0) {
                val digit2 = num2[j] - '0'
                resultDigits[j + i + 1] += digit1 * digit2
            }
        }
        for (i in resultDigits.lastIndex downTo 1) {
            resultDigits[i].let { v -> if (v > 0) resultDigits[i] = v % 10; resultDigits[i-1] += v / 10 }
        }
        val startIdx = resultDigits.indexOfFirst { it > 0 }
        return if (startIdx < 0) "0" else String(CharArray(resultDigits.size - startIdx) { '0' + resultDigits[it + startIdx] })
    }

    // 2444. Count Subarrays With Fixed Bounds
    fun countSubarrays(nums: IntArray, minK: Int, maxK: Int): Long {
        var count = 0L
        var minIdx = -1
        var maxIdx = -1
        var outOfBoundsIdx = -1
        nums.forEachIndexed { i, num ->
            if (num > maxK || num < minK) outOfBoundsIdx = i
            else {
                if (num == minK) minIdx = i
                if (num == maxK) maxIdx = i
                count += maxOf(0, minOf(minIdx, maxIdx) - outOfBoundsIdx)
            }
        }
        return count
    }

    // 1345. Jump Game IV
    fun minJumps(arr: IntArray): Int {
        val indicesByValue = arr.indices.groupByTo(hashMapOf()) { arr[it] }
        val visited = BooleanArray(arr.size).also { it[0] = true }
        val target = arr.lastIndex
        var jumpCount = 0
        val queue = ArrayDeque<Int>().apply { add(0) }
        while (queue.isNotEmpty()) {
            repeat(queue.count()) {
                val index = queue.removeFirst()
                if (index == target) return jumpCount
                indicesByValue.remove(arr[index])?.forEach { if (!visited[it]) { queue.add(it); visited[it] = true } }
                (index - 1).let { if (it >= 0 && !visited[it]) { queue.add(it); visited[it] = true } }
                (index + 1).let { if (it <= target && !visited[it]) { queue.add(it); visited[it] = true } }
            }
            ++jumpCount
        }
        return jumpCount
    }

    // 328. Odd Even Linked List
    fun oddEvenList(head: ListNode?): ListNode? {
        if (head?.next == null) return head
        val evenListHead = ListNode(0)
        var evenListTail = evenListHead
        var oddNode = head
        while (oddNode != null) {
            val evenNode = oddNode.next
            if (evenNode != null) {
                oddNode.next = evenNode.next
                evenListTail.next = evenNode
                evenListTail = evenNode
            } else {
                // oddNode.next = evenListHead.next
                evenListTail.next = null
            }
            oddNode = if (evenListTail.next != null) evenListTail.next else {
                oddNode.next = evenListHead.next
                null
            }
        }
        evenListTail.next = null
        return head
    }

    // 148. Sort List
    fun sortList(head: ListNode?): ListNode? {
        fun merge(left: ListNode, right: ListNode): ListNode {
            var leftTail: ListNode? = left
            var rightTail: ListNode? = right
            val acc = if (left.`val` <= right.`val`) {
                leftTail = left.next
                left
            } else {
                rightTail = right.next
                right
            }
            var tail = acc
            while (leftTail != null && rightTail != null) {
                if (leftTail.`val` <= rightTail.`val`) {
                    tail.next = leftTail
                    leftTail = leftTail.next
                } else {
                    tail.next = rightTail
                    rightTail = rightTail.next
                }
                tail = tail.next!!
            }
            tail.next = leftTail ?: rightTail
            return acc
        }

        fun sort(node: ListNode): ListNode {
            if (node.next == null) return node
            var slow: ListNode = node
            var fast: ListNode? = node.next
            while (fast?.next != null) {
                slow = slow.next!!
                fast = fast.next!!.next
            }
            val right = slow.next.also { slow.next = null }
            return merge(sort(node), sort(right!!))
        }
        return if (head?.next == null) head else sort(head)
    }

    val vowels = "AEIOUaeiou"

    // 1704. Determine if String Halves Are Alike
    fun halvesAreAlike(s: String): Boolean =
        (0 until s.length / 2).count { s[it] in vowels } == (s.length / 2 until s.length).count { s[it] in vowels }

    // 1539. Kth Missing Positive Number
    fun findKthPositive(arr: IntArray, k: Int): Int {
        val offset = arr.first()
        fun countMissing(index: Int): Int = arr[index] - offset - index
        if (k < offset) return k
        val missedInArray = countMissing(arr.lastIndex)
        val targetMisses = k - offset + 1
        if (k - targetMisses > missedInArray) return arr.last() + targetMisses - missedInArray
        var leftIdx = 0
        var rightIdx = arr.lastIndex
        while (leftIdx <= rightIdx) {
            val mid = (leftIdx + rightIdx) / 2
            if (countMissing(mid) < targetMisses) leftIdx = mid + 1 else rightIdx = mid - 1
        }
        return arr[leftIdx-1] + targetMisses - countMissing(leftIdx - 1)
    }

    // 2131. Longest Palindrome by Concatenating Two Letter Words
    fun longestPalindrome(words: Array<String>): Int {
        val wordCount = hashMapOf<String, Int>()
        words.forEach { wordCount.compute(it) { _, prev -> (prev?:0) + 1 } }
        var palindromeCount = 0
        var hasOddPalindrome = false
        wordCount.forEach { (word, count) ->
            palindromeCount += if (word[0] == word[1]) {
                hasOddPalindrome = hasOddPalindrome || count % 2 == 1
                count / 2 * 4
            } else 2 * minOf(count, wordCount[word.reversed()]?:0)
        }
        return palindromeCount + if (hasOddPalindrome) 2 else 0
    }

    // 621. Task Scheduler
    fun leastInterval(tasks: CharArray, n: Int): Int {
        val counts = IntArray(26).apply { tasks.forEach { ++this[it-'A'] } }
        val max = counts.max()!!
        val maxCount = counts.count { it == max }
        val numberOfBatches = max - 1
        val batchSize = n - maxCount + 1
        val numberOfSlots = numberOfBatches * batchSize
        val availableTasks = tasks.size - max * maxCount
        val idleTasks = maxOf(numberOfSlots - availableTasks, 0)
        return tasks.size + idleTasks
    }

    // 101. Symmetric Tree
    fun isSymmetric(root: TreeNode?): Boolean {
        fun areEqual(left: TreeNode?, right: TreeNode?): Boolean = when {
            left == null || right == null -> left == right
            left.`val` != right.`val` -> false
            else -> areEqual(left.left, right.right) && areEqual(right.left, left.right)
        }
        return root == null || areEqual(root.left, root.right)
    }

    // 2306. Naming a Company
    fun distinctNames(ideas: Array<String>): Long {
        val ideasByPrefix = Array(26){ hashSetOf<String>() }
        ideas.forEach { ideasByPrefix[it[0] - 'a'].add(it.substring(1)) }
        var namesCount = 0L
        for (i in 0 until ideasByPrefix.lastIndex) {
            val firstGroup = ideasByPrefix[i]
            for (j in i + 1 until ideasByPrefix.size) {
                val secondGroup = ideasByPrefix[j]
                val minGroup = if (firstGroup.size <= secondGroup.size) firstGroup else secondGroup
                val maxGroup = if (firstGroup.size <= secondGroup.size) secondGroup else firstGroup
                val ideaIntersection = minGroup.count { it in maxGroup }
                namesCount += 2 * (minGroup.size - ideaIntersection) * (maxGroup.size - ideaIntersection)
            }
        }
        return namesCount
    }

    // 918. Maximum Sum Circular Subarray
    fun maxSubarraySumCircular(nums: IntArray): Int {
        var totalSum = 0
        var maxSum: Int = Int.MIN_VALUE
        var curMax = 0
        var minSum: Int = Int.MAX_VALUE
        var curMin = 0
        for (x in nums) {
            curMax = maxOf(x, curMax + x)
            maxSum = maxOf(maxSum, curMax)
            curMin = minOf(x, curMin + x)
            minSum = minOf(minSum, curMin)
            totalSum += x
        }
        return if (maxSum > 0) maxOf(maxSum, totalSum - minSum) else maxSum
    }

    // 2187. Minimum Time to Complete Trips
    fun minimumTime(time: IntArray, totalTrips: Int): Long {
        var minPrediction = 1L
        var maxPrediction = Long.MAX_VALUE
        while (minPrediction < maxPrediction) {
            val prediction = minPrediction + (maxPrediction - minPrediction) / 2
            var trips = 0L
            for (i in time.indices) {
                trips += prediction / time[i]
                if (trips >= totalTrips) break
            }
            if (trips < totalTrips) minPrediction = prediction + 1 else maxPrediction = prediction
        }
        return minPrediction
    }

    // 112. Path Sum
    fun hasPathSum(root: TreeNode?, targetSum: Int): Boolean {
        fun calculateSum(node: TreeNode, accSum: Int): Boolean = (accSum + node.`val`).let { updatedAcc ->
            if (node.left == null && node.right == null) updatedAcc == targetSum
            else (node.left?.let  { calculateSum(it, updatedAcc) } ?: false) ||
                 (node.right?.let { calculateSum(it, updatedAcc) } ?: false)
        }
        return root != null && calculateSum(root, 0)
    }

    // 437. Path Sum III
    fun pathSum(root: TreeNode?, targetSum: Int): Int {
        fun calculatePartialSum(root: TreeNode?, acc: Long, cache: HashMap<Long, Int>): Int {
            if (root == null) return 0
            val currSum = acc + root.`val`
            val count = cache[currSum - targetSum]?:0
            cache.compute(currSum) { _, prev -> (prev?:0) + 1 }
            return count + calculatePartialSum(root.left, currSum, cache) + calculatePartialSum(root.right, currSum, cache)
                .also { cache.compute(currSum) { _, prev -> (prev?:0) - 1 } }
        }
        return calculatePartialSum(root, 0, hashMapOf<Long, Int>().apply { this[0L] = 1 })
    }

    // 875. Koko Eating Bananas
    fun minEatingSpeed(piles: IntArray, h: Int): Int {
        var minPrediction = 1
        var maxPrediction = piles.max()!!
        while (minPrediction < maxPrediction) {
            val prediction = (minPrediction + maxPrediction) / 2
            var time = 0
            for (pile in piles) {
                time += (pile - 1) / prediction + 1
                if (time > h) {
                    break
                }
            }
            if (time <= h) maxPrediction = prediction else minPrediction = prediction + 1
        }
        return minPrediction
    }

    // 543. Diameter of Binary Tree
    fun diameterOfBinaryTree(root: TreeNode?): Int {
        var diameter = 0
        fun height(node: TreeNode?): Int {
            if (node == null) return 0
            val leftHeight = height(node.left)
            val rightHeight = height(node.right)
            diameter = maxOf(diameter, leftHeight + rightHeight)
            return 1 + maxOf(leftHeight, rightHeight)
        }
        return maxOf(height(root), diameter)
    }

    // 700. Search in a Binary Search Tree
    fun searchBST(root: TreeNode?, `val`: Int): TreeNode? = when {
        root == null -> null
        `val` < root.`val` -> searchBST(root.left, `val`)
        `val` > root.`val` -> searchBST(root.right, `val`)
        else -> root
    }

    // 701. Insert into a Binary Search Tree
    fun insertIntoBST(root: TreeNode?, `val`: Int): TreeNode? {
        if (root == null) return TreeNode(`val`)
        var prevNode: TreeNode
        var node = root
        do {
            prevNode = node!!
            if (`val` < prevNode.`val`) node = prevNode.left else node = prevNode.right
        } while (node != null)
        TreeNode(`val`).let { if (`val` < prevNode.`val`) prevNode.left = it else prevNode.right = it }
        return root
    }

    // 108. Convert Sorted Array to Binary Search Tree
    fun sortedArrayToBST(nums: IntArray): TreeNode? {
        fun createBST(left: Int, right: Int): TreeNode {
            val mid = (left + right) / 2
            return TreeNode(nums[mid])
                .also { if (left < mid) it.left = createBST(left, mid - 1) }
                .also { if (right > mid) it.right = createBST(mid + 1, right) }
        }
        return when (nums.size ) {
            0 -> null
            1 -> TreeNode(nums[0])
            else -> createBST(0, nums.lastIndex)
        }
    }

    // 230. Kth Smallest Element in a BST
    fun kthSmallest(root: TreeNode?, k: Int): Int {
        val stack = ArrayDeque<TreeNode>()
        var leftToProcess = k
        var node = root
        do {
            while (node != null) {
                stack += node
                node = node.left
            }
            val polled = stack.removeLast()
            if (--leftToProcess == 0) return polled.`val`
            node = polled.right
        } while (node != null || stack.isNotEmpty())
        throw IllegalStateException("Not enough nodes")
    }

    // 109. Convert Sorted List to Binary Search Tree
    fun sortedListToBST(head: ListNode?): TreeNode? {
        fun getMiddle(node: ListNode): ListNode {
            var fast: ListNode? = node
            var slow: ListNode = node
            var prev: ListNode? = null
            while (fast?.next != null) {
                fast = fast.next!!.next
                prev = slow
                slow = slow.next!!
            }
            prev?.next = null
            return slow
        }
        if (head == null) return null
        if (head.next == null) return TreeNode(head.`val`)
        val middle = getMiddle(head)
        val root = TreeNode(middle.`val`)
        root.right = sortedListToBST(middle.next)
        middle.next = null
        root.left = sortedListToBST(head)
        return root
    }

    // 417. Pacific Atlantic Water Flow
    fun pacificAtlantic(heights: Array<IntArray>): List<List<Int>> {
        val n = heights.size
        val m = heights[0].size
        fun dfs(i: Int, j: Int, prev: Int, visited: Array<BooleanArray>) {
            if ((i in 0 until n) && (j in 0 until m) && !visited[i][j] && prev <= heights[i][j]) {
                visited[i][j] = true;
                dfs(i + 1, j, heights[i][j], visited)
                dfs(i - 1, j, heights[i][j], visited)
                dfs(i, j + 1, heights[i][j], visited)
                dfs(i, j - 1, heights[i][j], visited)
            }
        }
        val pacific = Array(n) { BooleanArray(m) }
        val atlantic = Array(n) { BooleanArray(m) }
        for (i in 0 until m) {
            dfs(0, i, heights[0][i], pacific)
            dfs(n - 1, i, heights[n - 1][i], atlantic)
        }
        for (i in 0 until n) {
            dfs(i, 0, heights[i][0], pacific)
            dfs(i, m - 1, heights[i][m - 1], atlantic)
        }
        val result = mutableListOf<List<Int>>()
        for (i in 0 until n) {
            for (j in 0 until m) {
                if (pacific[i][j] && atlantic[i][j]) {
                    result.add(listOf(i, j))
                }
            }
        }
        return result
    }

    // 367. Valid Perfect Square
    fun isPerfectSquare(num: Int): Boolean {
        val numLong = num.toLong()
        var from = 1L
        var to = 1000000L
        while (from <= to) {
            val mid = (from + to) / 2
            val square = mid * mid
            when {
                square > numLong -> to = mid - 1
                square < numLong -> from = mid + 1
                else -> return true
            }
        }
        return false
    }

    // 1385. Find the Distance Value Between Two Arrays
    fun findTheDistanceValue(arr1: IntArray, arr2: IntArray, d: Int): Int =
        arr1.count { element -> arr2.none { Math.abs(element - it) <= d } }

    // 210. Course Schedule II
    fun findOrder(numCourses: Int, prerequisites: Array<IntArray>): IntArray {
        val adjacent = Array(numCourses) { mutableListOf<Int>() }
        for ((course, prerequisite) in prerequisites) {
            adjacent[prerequisite].add(course)
        }
        val queue: Queue<Int> = ArrayDeque<Int>()
        val indegree = IntArray(numCourses)

        for (courses in adjacent) {
            for (i in courses.indices) {
                ++indegree[courses[i]]
            }
        }
        for (i in indegree.indices) { if (indegree[i] == 0) queue += i }

        val ans = IntArray(numCourses)
        var i = 0
        while (queue.isNotEmpty()) {
            val node = queue.remove()
            ans[i++] = node
            for (course in adjacent[node]) { if (--indegree[course] == 0) queue += course }
        }
        return if (i == numCourses) ans else intArrayOf()
    }

    // 815. Bus Routes
    fun numBusesToDestination(routes: Array<IntArray>, source: Int, target: Int): Int {
        val stopToRouteIndexes = hashMapOf<Int, ArrayList<Int>>()
        for (routeIndex in routes.indices) {
            for (busStop in  routes[routeIndex]) {
                stopToRouteIndexes.getOrPut(busStop) { ArrayList() }.add(routeIndex)
            }
        }
        val queue = ArrayDeque<Int>().apply { add(source) }
        val busStopsVisited = hashSetOf<Int>().apply { add(source) }
        val routesVisited = BooleanArray(routes.size)

        var count = 0
        while (queue.isNotEmpty()) {
            repeat(queue.size) {
                val stop = queue.removeFirst()
                if (stop == target) { return count }
                val buses = stopToRouteIndexes.getOrElse(stop) { emptyList<Int>() }
                for (bus in buses) {
                    if (!routesVisited[bus]) {
                        for (nextStop in routes[bus]) {
                            if (busStopsVisited.add(nextStop)) { queue.addLast(nextStop) }
                        }
                        routesVisited[bus] = true
                    }
                }
            }
            count++
        }
        return -1
    }

    // 986. Interval List Intersections
    fun intervalIntersection(firstList: Array<IntArray>, secondList: Array<IntArray>): Array<IntArray> {
        val n1 = firstList.size
        val n2 = secondList.size
        if (n1 == 0 || n2 == 0) return emptyArray()
        val ans = mutableListOf<IntArray>()
        var i = 0
        var j = 0
        while (i < n1 && j < n2) {
            val (startFirst, endFirst) = firstList[i]
            val (startSecond, endSecond) = secondList[j]
            if (startFirst <= endSecond && endFirst >= startSecond) {
                ans.add(intArrayOf(maxOf(startFirst, startSecond), minOf(endFirst, endSecond)))
            }
            if (endFirst < endSecond) i += 1 else j += 1
        }
        return ans.toTypedArray()
    }

    // 69. Sqrt(x)
    fun mySqrt(x: Int): Int {
        if (x == 0) { return 0 }
        var low = 1
        var high = x
        while (low <= high) {
            val mid = low + (high - low) / 2
            val div = x / mid
            when {
                mid > div -> high = mid - 1
                mid < div -> low = mid + 1
                else -> return mid
            }
        }
        return high
    }

    // 744. Find Smallest Letter Greater Than Target
    fun nextGreatestLetter(letters: CharArray, target: Char): Char {
        var low = 0
        var high = letters.lastIndex
        while (low <= high) {
            val mid = low + (high - low) / 2
            if (target < letters[mid]) {
                high = mid - 1
            } else {
                low = mid + 1
            }
        }
        return letters[low % letters.size]
    }
}

class QuadTreeNode(var `val`: Boolean, var isLeaf: Boolean) {
    var topLeft: QuadTreeNode? = null
    var topRight: QuadTreeNode? = null
    var bottomLeft: QuadTreeNode? = null
    var bottomRight: QuadTreeNode? = null
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
open class GuessGame(private val guess: Int) {
    fun guess(num:Int):Int = Integer.signum(guess - num)
}

// 374. Guess Number Higher or Lower
class GuessGameSolution(guess: Int) : GuessGame(guess) {
    fun guessNumber(n:Int):Int {
        var low = 1
        var high = n
        while (true) {
            val test = low + (high - low) / 2
            when(guess(test)) {
                -1 -> high = test - 1
                 1 -> low = test + 1
                else -> return test
            }
        }
    }
}

// 382. Linked List Random Node
class RandomLinkedListSolution(head: ListNode?) {
    private val arrayList = toArrayList(head)
    private val random = java.util.Random()
    private fun toArrayList(head: ListNode?): List<Int> {
        if (head == null) return emptyList()
        val result = ArrayList<Int>(10000)
        var node = head
        while(node != null) {
            result += node.`val`
            node = node.next
        }
        return result
    }

    fun getRandom(): Int = arrayList[random.nextInt(arrayList.size)]

}