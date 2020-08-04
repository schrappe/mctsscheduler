# Returns the last finish date of all tasks
def lastFinishMetric(currentSchedulePlan):
    constraintResource = 0
    resourceAllocation = []
    latestFinish = 0
    for task in currentSchedulePlan:
        taskFinish = task['finish']
        taskResource = task['resource']
        taskDuration = task['duration']
        if taskFinish > latestFinish:
            latestFinish = taskFinish
        # Finds the corresponding resource on the resource list and adds task duration to it
        foundResource = False
        for resource in resourceAllocation:
            if resource['resource'] == taskResource:
                resource['usage'] += taskDuration
                foundResource = True

        # If the resource has not been visited yet, include it on the list
        if not foundResource:
            resourceAllocation.append({'resource': taskResource, 'usage': taskDuration})

        # Finds the constraint resource (with highest allocation)
        constraint = max(resourceAllocation, key=lambda x:x['usage'])
        constraintResource = constraint['resource']
        constraintUsage = constraint['usage']

        # Finds the constraint occupation rate over the scheduling horizon
        occupationRate = constraintUsage / latestFinish

    return latestFinish, constraintResource, occupationRate




