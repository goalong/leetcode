# encoding: utf-8

class Solution1(object):
    def convert(self, s, numRows):
        ret = ''
        T = max((numRows - 1)*2, 1) # period of one zigzag
        for r in range(numRows):
            i = r
            while i < len(s):
                ret += s[i]
                # increase by an amount q that satisfies
                #  i + q = -i (mod T)  and  0 < q <= T
                i += T - (2*i % T)
        return ret


class Solution2(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1 or numRows >= len(s):
            return s
        l = [""] * numRows
        index, step = 0, 1
        for i in s:
            l[index] += i
            if index == 0:
                step = 1
            elif index == numRows - 1:
                step = -1
            index += step
        return ''.join(l)


s = Solution2()
r = s.convert("abc", 2)
print r


class Solution3():
    # @return an integer
    def lengthOfLongestSubstring(self, s):
        start = maxLength = 0
        usedChar = {}

        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            else:
                maxLength = max(maxLength, i - start + 1)

            usedChar[s[i]] = i

        return maxLength

    def lengthOfLongestSubstring2(self, s):
        """
        :type s: str
        :rtype: int
        """
        start, max_length = 0, 0
        memo = {}
        for i in range(len(s)):
            if s[i] in memo and start <= memo[s[i]]:
                start = memo[s[i]] + 1
            else:
                max_length = max(max_length, i - start + 1)
            memo[s[i]] = i
        return max_length

s = Solution3().lengthOfLongestSubstring2("tmmzuxt")
print s


class Solution7(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        is_positive = (x >= 0)
        str_x = str(abs(x))
        rs = str_x[::-1]
        if abs(x) > 0x7FFFFFFF:
            return 0
        if not is_positive:
            rs = -(int(rs))
        else:
            rs = int(rs)
        return rs
s = Solution7()
r = s.reverse(1534236469)
print r

class Solution8(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.strip()
        if len(str) == 0:
            return 0
        flag = ""
        if str[0] in ("+", "-"):
            flag = str[0]
            str = str[1:]

        rs = flag
        for i in str:
            if i.isdigit():
                rs += i
            else:
                break
        try:
            rs = int(rs)
            if -2 ** 31 < rs < 2 ** 31:
                return rs
            elif flag == "-":
                return -2**31
            else:
                return 2**31
        except:
            return 0

print Solution8().myAtoi("-2147483648")



class Solution14(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        index = 0
        try:
            while strs:
                if [i[index] == strs[0][index] for i in strs] == [True] * len(strs):
                    index += 1
                else:
                    break

        except Exception:
            pass
        if index == 0:
            return ''
        else:
            return strs[0][:index+1]
print Solution14().longestCommonPrefix([])


class Solution15(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        idea: 转化为2sum
        """
        data = {}
        rs = []
        nums.sort()
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            target = 0 - nums[i]
            for j in range(i + 1, len(nums)):
                if target - nums[j] in data:
                    rs.append((nums[i], nums[j], target - nums[j]))
                data[nums[j]] = j
            data = {}

        return list(set(tuple(rs)))   #remove duplicate

print Solution15().threeSum([0,0, 0, 0])

class Solution16(object):  #Todo, too slow, 278ms, beat 27%
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        memo = {}
        min = 0xffffff
        for i in range(len(nums)):
            if i >0 and nums[i] == nums[i-1]:
                continue
            new_target = target - nums[i]
            j, k = i+1, len(nums)-1
            while j < k:
                cur_sum = nums[i]+nums[j]+nums[k]
                close = abs(target-cur_sum)
                if close < min:
                    min, rs = close, nums[i]+nums[j]+nums[k]
                if cur_sum <= target:
                    j += 1
                else:
                    k -= 1
        return rs


class Solution17(object):
    def letterCombinations(self, digits):
        if not digits:
            return []
        results = ['']
        map = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}

        for digit in digits:
            results = [result + d for result in results for d in map[digit]]
            print results

        return results

print Solution17().letterCombinations('235')

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution19(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        if n == 1 and head.next is None:
            return []
        start = ListNode(None)
        start.next = head
        node, obj_node, n_node = head, start, head
        for i in range(n-1):
            n_node = n_node.next
        if n_node.next is None:
            return head.next
        while n_node.next:
            n_node = n_node.next
            obj_node = obj_node.next
        obj_node.next = obj_node.next.next
        return head

x = ListNode(1)
x.next = ListNode(2)
x.next.next = ListNode(3)
r = Solution19().removeNthFromEnd(x, 3)
print r.val


class Solution20(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        data = {"(": ")", "[": "]", "{": "}"}
        rs = []
        for i in s:
            if len(rs) >0 and rs[-1] in data.values():
                return False
            elif i in data.values() and len(rs) >0 and i == data.get(rs[-1]):
                rs.pop()
            else:
                rs.append(i)
        return rs == []

print Solution20().isValid("()")


class Solution21(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        idea: 小心处理边界
        """
        current= ListNode(None)
        head = current
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                current, l1 = current.next, l1.next
            else:
                current.next = l2
                current, l2 = current.next, l2.next
        if not l1 and not l2:
            pass
        elif not l1:
            current.next = l2
        else:
            current.next = l1
        return head.next


r = Solution21().mergeTwoLists(ListNode(None), ListNode(2))
print r.val


class Solution22(object):  #Todo,

    def generateParenthesis(self, n):
        ans = []
        def _generate(cur, left, right):
            if left > right:
                return
            if left == 0 and right == 0:
                ans.append(cur)
                return

            if left > 0:
                _generate(cur + '(', left - 1, right)

            if right > 0:
                _generate(cur + ')', left, right - 1)
        _generate('', n, n)
        return ans
print Solution22().generateParenthesis(3)


class Solution23(object):
    def merge(self, l1, l2):
        current= ListNode(None)
        head = current
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                current, l1 = current.next, l1.next
            else:
                current.next = l2
                current, l2 = current.next, l2.next
        if not l1 and not l2:
            pass
        elif not l1:
            current.next = l2
        else:
            current.next = l1
        return head.next
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if len(lists) == 0:
            return None
        if len(lists) == 1:
            return lists[0]
        mid = len(lists)/2
        left = self.mergeKLists(lists[:mid])
        right = self.mergeKLists(lists[mid:])
        return self.merge(right, left)

r = Solution23().mergeKLists([ListNode(None), ListNode(1)])
print r


class Solution24(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        node, node.next = self, head
        while node.next and node.next.next:
            node.next, node.next.next, node.next.next.next = node.next.next, node.next, node.next.next.next
            node = node.next.next
        return self.next

node1 = ListNode(1)
node3 = ListNode(3)
node3.next = ListNode(4)
node1.next = ListNode(2)
node1.next.next = node3
r = Solution24().swapPairs(node1)
print r


class Solution26(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        else:
            index = 0
            for i in range(1, len(nums)):
                if nums[i] != nums[index]:
                    index += 1
                    nums[index] = nums[i]
        return index+1

print Solution26().removeDuplicates([1,1, 1])


class Solution27(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if len(nums) < 1:
            return len(nums)
        index = -1
        for i in range(len(nums)):
            if nums[i] != val:
                index += 1
                nums[index] = nums[i]
        return index+1

print Solution27().removeElement([2,2], 2)


class Solution28(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        needle_len = len(needle)
        if needle_len == 0:
            return 0
        for i in xrange(len(haystack)):
            if haystack[i: i+needle_len] == needle:
                return i
        return -1

print Solution28().strStr("", "")


class Solution29(object): #TimeLimit Error
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if (dividend >0 and divisor >0) or (divisor <0 and dividend<0):
            flag = True
        else:
            flag = False
        left, div = abs(dividend), abs(divisor)
        pow = 1
        ans = 0
        while left >= div:
            left -= div
            ans += pow
            div += div
            pow += pow
            if left < div:
                div = abs(divisor)
                pow = 1
        return ans if flag else -ans
print Solution29().divide(-1999, -2)


class Solution31(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums)<2:
            return None
        i = len(nums) - 1
        while nums[i] <= nums[i - 1] and i > 0:
            i -= 1
        index = i - 1
        for j in range(len(nums) - 1, -1, -1):
            if nums[j] > nums[index]:
                nums[j], nums[index] = nums[index], nums[j]
                nums[index + 1:] = sorted(nums[index + 1:])
                return

print Solution31().nextPermutation([5,1,1])


class Solution34(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        low = 0
        high = len(nums)-1
        mid = (low+high)/2
        while low <= high:
            mid = (low+high)/2
            if nums[mid] > target:
                high = mid - 1
            elif nums[mid] < target:
                low = mid + 1
            else:
                break
        if low > high:
            return [-1, -1]
        start = end = mid
        while start >= 0 and nums[start] == target:
            start -= 1
        while end < len(nums) and nums[end] == target:
            end += 1
        if end - 1 == 0:
            return [start + 1, 0]
        else:
            return [start + 1, end - 1]


class Solution35(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) / 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                high = mid - 1
            else:
                low = mid + 1
        if nums[mid] > target:
            return mid
        elif nums[mid] < target:
            return mid+1


print Solution35().searchInsert([1,2,3,6], 4)


class Solution38(object):  #Todo too slow, only beat 13%
    """这道题的描述不太清晰，test case的意思是以1为起始的第n个，表达的却是以n为起始的第n个，damn"""
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        index = 1
        rs = "1"
        while index < n:
            rs = self._helper(rs)
            index += 1
        return rs

    def _helper(self, n):
        n = n + "$"
        i = 1
        times = 1
        rs = ''

        while i < len(n) and i >=1:
            if n[i-1] == n[i]:
                times += 1
            else:
                rs += '%d%s' % (times, n[i-1])
                times = 1
            i += 1
        return rs

print Solution38().countAndSay(2)



class Solution39(object):   # from others

    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res

    def dfs(self, nums, remain, index, path, res):
        print("remain: %d, index: %d, path: %s, res: %s" % (remain, index, path , res))
        if remain < 0:
            return  # backtracking
        if remain == 0:
            res.append(path)
            return
        for i in xrange(index, len(nums)):
            if nums[i] > remain:
                break
            self.dfs(nums, remain - nums[i], i, path + [nums[i]], res)

class Solution40(object):  #from others solution
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        rs = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], rs)
        return rs

    def dfs(self, nums, remain, index, path, rs):
        if remain < 0:
            return
        if remain == 0:
            rs.append(path)
            return
        for i in xrange(index, len(nums)):
            if nums[i] > remain:
                break
            if i-1>=index and nums[i] == nums[i-1]: #why ????
                continue
            self.dfs(nums, remain - nums[i], i+1, path + [nums[i]], rs)
    def combinationSum3(self, candidates, target):
        result = []
        def dfs(target, beg, ares, candidates):
            if target == 0:
                result.append(ares)
                return
            for i in xrange(beg, len(candidates)):
                if candidates[i] > target: break
                if i - 1 >= beg and candidates[i] == candidates[i - 1]:
                    continue
                dfs(target - candidates[i], i + 1, ares + [candidates[i]], candidates)

        candidates.sort()
        dfs(target, 0, [], candidates)
        return result

print "la"
print Solution40().combinationSum2([10,1,2,7,6,1,5, 2,4, 2, 2], 8)


class Solution46(object):  #from others
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        return [[n] + p for i,n in enumerate(nums) for p in self.permute(nums[:i]+nums[i+1:])] or [[]]
print Solution46().permute([1,1, 2])


class Solution47(object):    #from others
    def permuteUnique(self, nums):
        ans = [[]]
        for n in nums:
            new_ans = []
            for l in ans:
                for i in xrange(len(l) + 1):
                    new_ans.append(l[:i] + [n] + l[i:])
                    if i < len(l) and l[i] == n:     #confuse ???
                        break  # handles duplication
            ans = new_ans
        return ans
print Solution47().permuteUnique([1,1, 2,3])


class Solution48(object):
    def rotate(self, matrix):  # from https://discuss.leetcode.com/topic/6796/a-common-method-to-rotate-the-image
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        matrix.reverse()
        for i in xrange(len(matrix)):
            for j in xrange(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        print matrix

print Solution48().rotate([[1,2,3], [4,5,6], [7,8,9]])


class Solution49(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        memo = {}
        for s in strs:
            s_sorted = "".join(sorted(s))
            if s_sorted in memo:
                memo[s_sorted].append(s)
            else:
                memo[s_sorted] = [s]
        return memo.values()

print Solution49().groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])



class Solution50(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        positive = (n > 0)
        n = abs(n)

        def helper(x, n):
            if n == 0:
                return 1
            if n == 1:
                return x
            if n > 1:
                if n % 2 == 1:
                    return helper(x * x, n / 2) * x
                else:
                    return helper(x * x, n / 2)

        rs = helper(x, n)
        return rs if positive else 1 / rs


print Solution50().myPow(1.001, 50)


class Solution53(object):
    def maxSubArray(self, nums):    #Time Limit Error
        """
        :type nums: List[int]
        :rtype: int
        """
        j, i = 0, 0
        max = nums[0]
        while i < len(nums):
            t_sum = sum(nums[j:i + 1])
            temp = j
            while temp < i+1:
                temp_sum = sum(nums[temp:i+1])
                if temp_sum > t_sum:
                    t_sum = temp_sum
                temp += 1
            if t_sum > max:
                max = t_sum
            i += 1
        return max

    def maxSubArray2(self, nums):   # from others
        rs, current_sum = nums[0], 0
        for i in nums:
            current_sum += i
            rs, current_sum = max(rs, current_sum), max(0, current_sum)
        return rs

print Solution53().maxSubArray2([-2,1,-3,4,-1,2,1,-5,4])


class Solution54(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        rs = []
        while matrix:
            rs += matrix.pop(0)
            if matrix and matrix[0]:
                for row in matrix:
                    rs.append(row.pop())
            if matrix and matrix[0]:
                rs += matrix.pop()[::-1]
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    rs.append(row.pop(0))
        return rs


print Solution54().spiralOrder(
    [[7], [9], [6]])



class Solution55(object):
    def canJump(self, nums):    # from others
        """
        :type nums: List[int]
        :rtype: bool
        """
        m = 0
        for i, n in enumerate(nums):
            if i > m:  # 前面的i-1个位置最远只能跳到第i－1位，后面的都无法到达了
                return False
            m = max(m, i + n)
        return True




print Solution55().canJump([3,2,1,0,4])



class Solution56(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        rs = []
        intervals.sort(key=lambda x: x.start)
        for i in intervals:
            if not rs or i.start > rs[-1].end:
                rs.append(i)
            elif i.end > rs[-1].end:
                rs[-1].end = i.end
        return rs

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e


class Solution57(object):
    def insert(self, intervals, newInterval):    #original, slow and stupid
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        rs = [newInterval]
        if not intervals:
            return rs
        for interval in intervals:
            if interval.end < rs[-1].start:  # [(1,5)] (6,8)
                rs.insert(-1, interval)
            elif interval.start <= rs[-1].start <= rs[-1].end <= interval.end:  # 1, 7 (2,3)
                rs[-1] = interval
            elif rs[-1].start <= interval.start <= interval.end <= rs[-1].end:  # 1, 5 (0,6)
                pass
            elif newInterval.end > interval.end >= newInterval.start >= interval.start:  # 1,5  (2, 8)
                rs[-1].start = interval.start
            elif interval.end >= rs[-1].end >= interval.start >= rs[-1].start:    # 3, 5 6, 9    4, 7
                rs[-1].end = interval.end
            elif interval.start > rs[-1].end:  #5,8  1,3
                rs.append(interval)
        return rs

    def insert2(self, intervals, newInterval):    # can't pass, Interval not Implement, leetcode bug
        left = [i for i in intervals if i.end < newInterval.start]
        right = [i for i in intervals if i.start > newInterval.end]
        if not left + right == intervals:
            start = min(intervals[len(left)].start, newInterval.start)
            end = max(intervals[~len(right)].end, newInterval.end)
            return left + [Interval(start, end)] + right
        return left + right

print Solution57().insert([Interval(0,10)], Interval(2,2))


class Solution58(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        str_list = s.split()
        return 0 if not str_list else len(str_list[-1])

print Solution58().lengthOfLastWord("a ")


class Solution59(object):
    def generateMatrix(self, n):   # from https://discuss.leetcode.com/topic/19130/4-9-lines-python-solutions
        """
        :type n: int
        :rtype: List[List[int]]
        """
        rs = []
        low = n*n + 1
        while low > 1:
            low, high = low - len(rs), low
            rs = [range(low, high)] + zip(*rs[::-1])
        return rs

    def generateMatrix2(self, n):
        rs = [[0 for a in range(n)] for b in range(n)]
        positions = [[(i, j) for j in range(n)] for i in range(n)]
        num = 1
        while positions:
            first_row = positions.pop(0)
            for x,y in first_row:
                print x,y, num
                rs[x][y] = num
                num += 1
            positions = zip(*positions)[::-1]
        return rs
print Solution59().generateMatrix2(3)


class Solution60(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        from math import factorial
        origin = range(1, n+1)
        rs = []
        extra = k
        i = 1
        while i < n -1:
            fact = factorial(n - i)
            index = extra // fact
            extra = extra % fact
            index = index -1 if extra == 0 else index
            rs.append(origin.pop(index))
            i+=1
        if extra == 1:
            rs.extend(origin)
        else:
            rs.extend(origin[::-1])
        return ''.join(str(e) for e in rs)

        # first = k // factorial(n-1) # index of first element in origin
        # extra = k % factorial(n-1)
        #
        # rs.append(origin.pop(first))
        #
        #
        # second = extra // factorial(n-2) # index of second element in origin
        # rs.append(origin.pop(second))
        #
        # extra = extra % factorial(n-2)
        #
        #
        # third = extra // factorial(n-3)
        # rs.append(origin.pop(third))

print(Solution60().getPermutation(3, 6))

class Solution61(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if k == 0 or not head:
            return head
        len = 0
        n = lastNode = head
        while n:
            len += 1
            lastNode = n
            n = n.next
        if k % len == 0:
            return head
        newHead = newTail = head
        for i in range((len - k) % len):
            newTail = newHead
            newHead = newHead.next
        lastNode.next = head
        newTail.next = None
        return newHead


class Solution62(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        rs = [[1] * n for x in range(m)]
        for x in range(1,m):
            for y in range(1, n):
                rs[x][y] = rs[x-1][y] + rs[x][y-1]  #每个位置的唯一路径等于它上边和左边位置的唯一路径的和
        return rs[m-1][n-1]

print(Solution62().uniquePaths(2, 2))


class Solution63(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        rs = [[0] * n for x in range(m)]
        for i in range(m):
            if obstacleGrid[i][0]:
                break
            rs[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j]:
                break
            rs[0][j] = 1

        for x in range(1, m):
            for y in range(1, n):
                if not obstacleGrid[x][y]:
                    rs[x][y] = rs[x - 1][y] + rs[x][y - 1]
        return rs[m - 1][n - 1]

print(Solution63().uniquePathsWithObstacles([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
]))




class Solution64(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = len(grid[0])
        rs = grid
        for i in range(1,m):
            rs[i][0] += rs[i-1][0]  #边线上的位置的最小sum等于该位置的值加前一个位置的最小sum

        for j in range(1,n):
            rs[0][j] += rs[0][j-1]

        for x in range(1,m):
            for y in range(1, n):
                rs[x][y] += min(rs[x-1][y], rs[x][y-1])  #该位置的最小sum等于该位置的值加左边和上边位置的最小sum中较小的一个
        return rs[m-1][n-1]

print(Solution64().minPathSum([[1,2,3], [1,1,1], [1,2,3]]))


class Solution66(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        rs = []
        extra = 0
        digits[-1] += 1
        while digits:
            cur_last = digits.pop()
            item = cur_last + extra
            if item > 9:
                rs.append(0)
                extra = 1
            else:
                rs.append(item)
                extra = 0
        if extra == 1:
            rs.append(1)
        return rs[::-1]

print(Solution66().plusOne([9,9]))


class Solution67(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        rs = []
        extra = 0
        list_a, list_b = list(a), list(b)
        while list_a or list_b:
            last_a = list_a.pop() if list_a else 0
            last_b = list_b.pop() if list_b else 0
            item = int(last_a) + int(last_b) + extra
            if item == 2:
                rs.append(0)
                extra = 1
            elif item == 3:
                rs.append(1)
                extra = 1
            else:
                rs.append(item)
                extra = 0
        if extra == 1:
            rs.append(1)
        return "".join([str(i) for i in rs[::-1]])

print(Solution67().addBinary("1111", "1111"))


class Solution69(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0 or x == 1:
            return x
        left, right = 0, x
        while left <= right:
            mid = (left+right)//2
            if mid**2 <= x <(mid+1)**2:
                return mid
            elif mid**2 < x:
                left = mid + 1
            else:
                right = mid -1


print(Solution69().mySqrt(1695165231))

