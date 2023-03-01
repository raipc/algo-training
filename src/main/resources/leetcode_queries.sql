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

-- 1141. User Activity for the Past 30 Days I
SELECT activity_date AS day, COUNT(DISTINCT user_id) AS active_users
FROM Activity
WHERE activity_date < '2019-07-28' AND DATEDIFF('2019-07-27', activity_date) < 30
GROUP BY activity_date

-- 1693. Daily Leads and Partners
SELECT date_id, make_name, COUNT(DISTINCT lead_id) AS unique_leads, COUNT(DISTINCT partner_id) AS unique_partners
FROM DailySales
GROUP BY date_id, make_name

-- 1729. Find Followers Count
SELECT user_id, COUNT(*) AS followers_count
FROM Followers
GROUP BY user_id
ORDER BY user_id

-- 586. Customer Placing the Largest Number of Orders
SELECT customer_number
FROM Orders
GROUP BY customer_number
ORDER BY COUNT(*) DESC
LIMIT 1

-- 511. Game Play Analysis I
SELECT player_id, MIN(event_date) AS first_login
FROM Activity
GROUP BY player_id

-- 1890. The Latest Login in 2020
SELECT user_id, MAX(time_stamp) AS last_stamp
FROM Logins
WHERE YEAR(time_stamp) = '2020'
GROUP BY user_id

-- 1741. Find Total Time Spent by Each Employee
SELECT event_day AS day, emp_id, SUM(out_time - in_time) AS total_time
FROM Employees
GROUP BY emp_id, event_day

-- 1407. Top Travellers
SELECT name, IFNULL(SUM(distance), 0) AS travelled_distance
FROM Users LEFT JOIN Rides ON Users.id = Rides.user_id
GROUP BY Users.id
ORDER BY travelled_distance DESC, name ASC

-- 1393. Capital Gain/Loss
SELECT stock_name, SUM(price * IF(operation = 'Buy', -1, 1)) capital_gain_loss
FROM Stocks
GROUP BY stock_name

-- 1158. Market Analysis I
SELECT user_id AS buyer_id, join_date, IFNULL(COUNT(order_id), 0) AS orders_in_2019
FROM Users LEFT JOIN Orders ON Users.user_id = Orders.buyer_id AND YEAR(order_date) = '2019'
GROUP BY Users.user_id

-- 182. Duplicate Emails
SELECT email
FROM Person
GROUP BY email
HAVING COUNT(*) > 1

-- 1050. Actors and Directors Who Cooperated At Least Three Times
SELECT actor_id, director_id
FROM ActorDirector
GROUP BY actor_id, director_id
HAVING COUNT(*) >= 3

-- 1587. Bank Account Summary II
SELECT name, SUM(amount) AS balance
FROM Users JOIN Transactions USING(account)
GROUP BY account
HAVING balance > 10000

-- 1084. Sales Analysis III
SELECT product_id, product_name
FROM Product JOIN Sales USING (product_id)
GROUP BY product_id
HAVING SUM(IF(sale_date BETWEEN '2019-01-01' AND '2019-03-31', 0, 1)) = 0