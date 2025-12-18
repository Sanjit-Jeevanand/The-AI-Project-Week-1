# 2 Pointer Problem

nums = [0,1,2,3,4,5]

left = 0
right = len(nums) - 1

while left < right:
    nums[left], nums[right] = nums[right], nums[left]
    left += 1
    right -= 1

print(nums)

def hasDuplicate(self, nums: List[int]) -> bool:
        arr = {}
        for i in range(len(nums)):
            arr.append(nums[i])
        if len(arr) == len(nums):
            return True
        return False