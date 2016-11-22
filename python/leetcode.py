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


r = Solution21().mergeTwoLists(ListNode(1), ListNode(2))
print r.val


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