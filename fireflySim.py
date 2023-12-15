# Code for simulating a flock of birds/ herd of bison
import pygame
# Used for calculations
import numpy as np

class Boid:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

# Setting up view window
width = 1434
height = 717


def updateBoid(boid: Boid, allBoids: list[Boid], cursorPosition):
    # CHANGE THESE - Radius
    seperationRadius = 44
    alignmentRadius = 30
    cohesionRadius = 26

    # CHANGE THESE - Weights
    seperationWeight = 2.1
    alignmentWeight = 0.7
    cohesionWeight = 0.7

    # CHANGE THIS - When hit wall, lose energy
    dampiningFactor = 0.7


    seperationNeighbors = []
    alignmentNeighbors = []
    cohesionNeighbors = []

    # Get the neighbors of the current boid
    for other in allBoids:
        if (np.linalg.norm(boid.position - other.position) <= seperationRadius and np.linalg.norm(boid.position - other.position) >= (-1 * seperationRadius) and boid != other):
            seperationNeighbors += [other]
        
        if (np.linalg.norm(boid.position - other.position) <= alignmentRadius and np.linalg.norm(boid.position - other.position) >= (-1 * alignmentRadius) and boid != other):
            alignmentNeighbors += [other]
    
        if (np.linalg.norm(boid.position - other.position) <= cohesionRadius and np.linalg.norm(boid.position - other.position) >= (-1 * cohesionRadius) and boid != other):
            cohesionNeighbors += [other]


    # Calculate Seperation, Alignment, and Cohesion
    currSeperation = calculateSeperation(boid, seperationNeighbors)
    currAlignment = calculateAlignment(boid, alignmentNeighbors)
    currCohesion = calculateCohesion(boid, cohesionNeighbors)


    # Steering
    maxForce = 0.1
    totalSteering = (seperationWeight * currSeperation) + (alignmentWeight * currAlignment) + (cohesionWeight * currCohesion)
    totalSteering = np.clip(totalSteering, -1 * maxForce, maxForce)

    if (boid.position[0] + totalSteering[0] < 0 or boid.position[0] + totalSteering[0] > width or boid.position[1] + totalSteering[1] < 0 or boid.position[1] + totalSteering[1] > height):
        boid.velocity = boid.velocity * -1 * dampiningFactor
        boid.position += boid.velocity
    else:
        cursor_force = cursorPosition - boid.position
        totalSteering += cursor_force

        max_cursor_force = 1.0
        totalSteering = np.clip(totalSteering, -max_cursor_force, max_cursor_force)

        # Get random max speed value
        randomMaxSpeed = np.random.uniform(3.0, 21.0)
        maxSpeed = 14.0  # Experiment with this value
        tempVelocity = boid.velocity + totalSteering

        boid.velocity = np.clip(tempVelocity, -randomMaxSpeed, randomMaxSpeed)
        boid.position += boid.velocity
    

def calculateSeperation(boid: Boid, neighbors: list[Boid]):
    seperationForce = np.zeros_like(boid.position)
    if (len(neighbors) <= 0):
        return seperationForce
    
    for each in neighbors:
        distanceToNeighbor = boid.position - each.position
        # np.linalg.norm() takes in a vector and calculates the magnitude of the hypotenus of that vector
        seperationForce += (distanceToNeighbor / np.linalg.norm(distanceToNeighbor))
    
    return seperationForce


def calculateAlignment(boid: Boid, neighbors: list[Boid]):
    if (len(neighbors) <= 0):
        return np.zeros_like(boid.position)
    
    totalVelocity = 0
    for neighborBoid in neighbors:
        totalVelocity += neighborBoid.velocity
    avgVelocity = np.mean(totalVelocity)

    alignmentForce = avgVelocity - boid.velocity
    return alignmentForce



def calculateCohesion(boid: Boid, neighbors: list[Boid]):
    if (len(neighbors) <= 0):
        return np.zeros_like(boid.position)

    totalNeighborPosition = 0
    for neighborBoid in neighbors:
        totalNeighborPosition += neighborBoid.position
    avgPosition = np.mean(totalNeighborPosition)

    cohesionForce = avgPosition - boid.position
    return cohesionForce

def drawFlies(screen, boids: list[Boid]):
    for curr in boids:
        pygame.draw.circle(screen, (255, 255, 0), (curr.position[0], curr.position[1]), 3)


def main():
    # Initialize pygame (2d simulation library)
    pygame.init()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Firefly Sim")



    bayouImage = pygame.image.load("bayou4.jpg")
    bayouImage = pygame.transform.scale(bayouImage, (1434, 717))

    # pygame.mouse.set_visible(False)

    # Number of boids in sim
    boidCount = 70
    boids = []
    for i in range(boidCount):
        boids += [Boid(position=np.random.rand(2) * np.array([width, height]), velocity=(np.random.rand(2) - 0.5) * 2)]


    # Main loop, handles events and updates sim state
    while True:
        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                pygame.quit()

        cursor_position = np.array(pygame.mouse.get_pos())

        # Black Background
        screen.fill((0, 0, 0))

        screen.blit(bayouImage, (0, 0))
        # screen.blit(honeyPotCursor, cursor_position)

        # Update all boids
        for currBoid in boids:
            updateBoid(currBoid, boids, cursor_position)

        drawFlies(screen, boids)

        # Updates display
        pygame.display.flip()

        pygame.time.Clock().tick(50)


# Run main method automatically
if __name__ == "__main__":
    main()


