# MST 함수
import pandas as pd

parent = {}
rank = {}


# 정점을 독립적인 집합으로 만든다.
def make_set(v):
    parent[v] = v
    rank[v] = 0


# 해당 정점의 최상위 정점을 찾는다.
def find(v):
    if parent[v] != v:
        parent[v] = find(parent[v])

    return parent[v]


# 두 정점을 연결한다.
def union(v, u):
    root1 = find(v)
    root2 = find(u)

    if root1 != root2:
        # 짧은 트리의 루트가 긴 트리의 루트를 가리키게 만드는 것이 좋다.
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2

            if rank[root1] == rank[root2]:
                rank[root2] += 1


def kruskal(graph):
    for v in graph['vertices']:
        make_set(v)

    mst = []

    edges = graph['edges']
    edges.sort()

    for edge in edges:
        weight, v, u = edge

        if find(v) != find(u):
            union(v, u)
            mst.append(edge)

    return mst

def mst_result(data):

    mst = data[['Weight','Source','Target']]

    mst['Weight'] = 1 / mst['Weight']

    tuples = [tuple(x) for x in mst.values]
    vertice = []
    s = list(mst['Source'].unique())
    t = list(mst['Target'].unique())
    vertice.extend(s)
    vertice.extend(t)
    vertices = list(set(vertice))
    graph = {'vertices': vertices, 'edges': tuples}

    result = kruskal(graph)

    Source = []
    Target = []
    Weight = []

    for i, j, k in result:
        Source.append(j)
        Target.append(k)
        Weight.append(i)

    G = pd.DataFrame([Source, Target, Weight]).T
    G.columns = ['Source', 'Target', 'Weight']

    G['Weight'] = 1 / G['Weight']

    return G