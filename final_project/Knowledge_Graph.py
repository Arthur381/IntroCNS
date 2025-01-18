import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import networkx as nx
import matplotlib.pyplot as plt

# 定义CIFAR-10的类别
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
classw = ["deer", "dog", "frog", "horse", "ship", "truck"]

# SPARQL 端点
sparql_endpoint = "http://dbpedia.org/sparql"

# 构建知识图谱
def build_knowledge_graph(classes, max_neighbors=5):
    """
    根据CIFAR-10的类别，从DBpedia构建知识图谱
    """
    graph = nx.Graph()
    sparql = SPARQLWrapper(sparql_endpoint)

    for cls in classes:
        print(f"Fetching neighbors for {cls}...")
        
        # SPARQL 查询语句
        query = f"""
        SELECT ?related ?relatedLabel WHERE {{
          ?concept rdfs:label "{cls}"@en .
          ?concept dbo:wikiPageWikiLink ?related .
          ?related rdfs:label ?relatedLabel .
          FILTER (lang(?relatedLabel) = 'en')
        }}
        LIMIT {max_neighbors}
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            
            for result in results["results"]["bindings"]:
                related = result["relatedLabel"]["value"]
                graph.add_edge(cls, related, weight=1.0)

        except Exception as e:
            print(f"Error fetching data for {cls}: {e}")

    return graph


def visualize_graph(graph):
    """
    可视化知识图谱
    """
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 12))
    nx.draw(
        graph, pos, with_labels=True, 
        node_size=3000, node_color="lightblue", font_size=10, font_weight="bold"
    )
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title("Knowledge Graph Visualization", fontsize=16)
    plt.show()


def get_related_concepts(graph, concept, threshold=0.5):
    """
    获取与指定概念相关的节点及权重
    """
    if concept not in graph:
        print(f"Concept '{concept}' not in graph.")
        return {}
    
    neighbors = {}
    for neighbor in graph.neighbors(concept):
        weight = graph[concept][neighbor]['weight']
        if weight >= threshold:
            neighbors[neighbor] = weight
    
    return neighbors

# 构建图谱
knowledge_graph = build_knowledge_graph(classw)

# 打印图谱信息
print("Knowledge Graph Nodes:", knowledge_graph.nodes())
print("Knowledge Graph Edges:", knowledge_graph.edges(data=True))

# 可视化知识图谱
visualize_graph(knowledge_graph)

# 示例: 查询与 "dog" 相关的概念
related_concepts = get_related_concepts(knowledge_graph, "dog")
print("Related concepts to 'dog':", related_concepts)
