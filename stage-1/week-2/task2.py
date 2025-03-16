import math


class Unit:
    def __init__(self, name, position):
        self.name = name
        self.position = position


class Enemy(Unit):
    def __init__(self, name, position, vector):
        super().__init__(name, position)
        self.life_points = 10
        self.vector = vector
        self.is_dead = False

    def move(self):
        if self.is_dead:
            return
        self.position[0] += self.vector[0]
        self.position[1] += self.vector[1]

    def be_attacked(self, attack_points):
        if self.is_dead:
            return
        self.life_points -= attack_points
        if self.life_points <= 0:
            self.is_dead = True


class BasicTower(Unit):
    def __init__(self, name, position):
        super().__init__(name, position)
        self.attack_points = 1
        self.attack_range = 2

    def is_enemy_in_range(self, enemy):
        if enemy.is_dead:
            return False
        distance = math.dist(self.position, enemy.position)
        return distance <= self.attack_range

    def attack(self, enemy):
        if enemy.is_dead:
            return
        enemy.be_attacked(self.attack_points)


class AdvancedTower(BasicTower):
    def __init__(self, name, position):
        super().__init__(name, position)
        self.attack_points = 2
        self.attack_range = 4

    def is_enemy_in_range(self, enemy):
        return super().is_enemy_in_range(enemy)

    def attack(self, enemy):
        super().attack(enemy)


###################################################

E1 = Enemy("E1", [-10, 2], [2, -1])
E2 = Enemy("E2", [-8, 0], [3, 1])
E3 = Enemy("E3", [-9, -1], [3, 0])

T1 = BasicTower("T1", [-3, 2])
T2 = BasicTower("T2", [-1, -2])
T3 = BasicTower("T3", [4, 2])
T4 = BasicTower("T4", [7, 0])

A1 = AdvancedTower("A1", [1, 1])
A2 = AdvancedTower("A2", [4, -3])

###################################################

turns = 10
enemies = [E1, E2, E3]
towers = [T1, T2, T3, T4, A1, A2]

for i in range(turns):
    for enemy in enemies:
        enemy.move()

    for tower in towers:
        for enemy in enemies:
            is_in_range = tower.is_enemy_in_range(enemy)
            if is_in_range and not enemy.is_dead:
                tower.attack(enemy)

    if i == turns - 1:
        for enemy in enemies:
            print(enemy.name, enemy.position, enemy.life_points)
