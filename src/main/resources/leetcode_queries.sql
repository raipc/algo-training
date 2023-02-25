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

-- 196. Delete Duplicate Emails
DELETE p1 FROM Person p1, Person p2 WHERE p1.email = p2.email AND p1.id > p2.id

-- 1667. Fix Names in a Table
SELECT user_id, CONCAT(UPPER(LEFT(name, 1)), LOWER(SUBSTRING(name, 2))) as name
FROM Users
ORDER BY user_id

-- 1484. Group Sold Products By The Date
SELECT sell_date,
       COUNT(DISTINCT product) AS num_sold,
       GROUP_CONCAT(DISTINCT product ORDER BY product) AS products
FROM Activities
GROUP BY sell_date
ORDER BY sell_date

-- 1527. Patients With a Condition
SELECT patient_id, patient_name, conditions
FROM Patients
WHERE conditions REGEXP '\\bDIAB1'

-- 1965. Employees With Missing Information
SELECT employee_id
FROM (
    SELECT e.employee_id
    FROM Employees e LEFT JOIN Salaries s USING(employee_id)
    WHERE s.salary IS NULL
    UNION
    SELECT s.employee_id
    FROM Employees e RIGHT JOIN Salaries s USING(employee_id)
    WHERE e.name IS NULL) t
ORDER BY 1

-- 1795. Rearrange Products Table
SELECT product_id, 'store1' AS store, store1 AS price
FROM Products
WHERE store1 IS NOT NULL
UNION ALL
SELECT product_id, 'store2' AS store, store2 AS price
FROM Products
WHERE store2 IS NOT NULL
UNION ALL
SELECT product_id, 'store3' AS store, store3 AS price
FROM Products
WHERE store3 IS NOT NULL

-- 608. Tree Node
SELECT id, 'Root' AS type
FROM Tree
WHERE p_id IS NULL
UNION
SELECT DISTINCT t1.id, 'Inner' AS type
FROM Tree t JOIN Tree t1 ON t.p_id = t1.id
WHERE t1.p_id IS NOT NULL
UNION
SELECT t.id, 'Leaf' AS type
FROM Tree t LEFT JOIN Tree t1 ON t1.p_id = t.id
WHERE t1.id IS NULL AND t.p_id IS NOT NULL

-- 176. Second Highest Salary
SELECT MAX(salary) AS SecondHighestSalary
FROM EMPLOYEE
WHERE SALARY < (SELECT MAX(salary) FROM EMPLOYEE)

-- 175. Combine Two Tables
SELECT firstName, lastName, city, state
FROM Person LEFT JOIN Address USING (personId)

-- 1581. Customer Who Visited but Did Not Make Any Transactions
SELECT customer_id, COUNT(*) AS count_no_trans
FROM Visits LEFT JOIN Transactions USING(visit_id)
WHERE transaction_id IS NULL
GROUP BY customer_id

-- 1148. Article Views I
SELECT DISTINCT viewer_id AS id
FROM Views
WHERE viewer_id = author_id
ORDER BY viewer_id ASC

-- 197. Rising Temperature
SELECT t.id
FROM Weather t JOIN Weather t1 ON t1.recordDate = DATE_SUB(t.recordDate, INTERVAL 1 DAY)
WHERE t.temperature > t1.temperature

-- 607. Sales Person
SELECT name
FROM SalesPerson
WHERE sales_id NOT IN (
    SELECT sales_id
    FROM Company JOIN Orders USING (com_id)
    WHERE name='RED'
)