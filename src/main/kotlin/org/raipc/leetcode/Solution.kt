package org.raipc.leetcode

import java.lang.StringBuilder
import java.util.*

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
}

class ListNode(var `val`: Int) {
    var next: ListNode? = null

    override fun toString(): String {
        return "val=${`val`}, next ${next}"
    }
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