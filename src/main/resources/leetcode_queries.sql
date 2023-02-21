-- 595. Big Countries
SELECT name, population, area
FROM World
WHERE area >= 3000000 OR population >= 25000000

-- 1757. Recyclable and Low Fat Products
SELECT product_id
FROM Products
WHERE low_fats = 'Y' AND recyclable = 'Y'

-- 584. Find Customer Referee
SELECT name
FROM Customer
WHERE referee_id is null OR referee_id <> 2

-- 183. Customers Who Never Order
SELECT c.name as Customers
FROM Customers c LEFT JOIN Orders o ON c.id = o.customerId
WHERE o.id is null

-- 1873. Calculate Special Bonus
SELECT employee_id, IF(mod(employee_id, 2) = 1 AND LEFT(name, 1) <> 'M', salary, 0) AS bonus
FROM Employees
ORDER BY employee_id

-- 627. Swap Salary
UPDATE Salary SET sex = IF(sex='f','m','f')