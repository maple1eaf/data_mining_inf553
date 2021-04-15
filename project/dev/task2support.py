DECIMAL_ACCURACY = 0.00000000001

class Vertex: 
    def __init__(self, vtx_name): 
        self.vertex = vtx_name 
        self.links = {} # {vertex_name: weight, ...}
        self.degree = None
    
    def addLink(self, other, weight):
        other_name = other.vertex
        self.links[other_name] = weight
    
    def removeLink(self, vtx_name):
        removed_weight = self.links.pop(vtx_name, None)
        if self.degree != None and removed_weight != None:
            self.degree -= 1
        return removed_weight

    def freshDegree(self):
        self.degree = len(self.links)
    
    def getDegree(self):
        return self.degree
    
    def __str__(self):
        return 'vertex %s links %s' % (self.vertex, self.links)

class UndirectedGraph:
    def __init__(self, vertices=None, edges=None):
        self.vertices = {}
        self.num_vertex = 0 # the number of vertices
        self.num_edge = 0 # the number of edges
        self.betweenness = None
        self.groups = None
        self.vertices_copy = vertices
        self.edges_copy = edges

        if vertices != None:
            for v in vertices:
                self.addVertex(v)
        if edges != None:
            for e in edges:
                self.addEdge(e[0], e[1], e[2])

    def addVertex(self, vtx_name):
        new_vtx = Vertex(vtx_name)
        self.vertices[vtx_name] = new_vtx
        self.num_vertex += 1
        return new_vtx

    def addEdge(self, frm, to, weight):
        # frm & to - name of the Node object
        if frm not in self.vertices:
            self.addVertex(frm)
        if to not in self.vertices:
            self.addVertex(to)
        frm_vtx = self.vertices[frm]
        to_vtx = self.vertices[to]
        frm_vtx.addLink(to_vtx, weight)
        to_vtx.addLink(frm_vtx, weight)
        self.num_edge += 1
    
    def removeEdge(self, edge):
        # edge - (vname1, vname2)
        vname1 = edge[0]
        vname2 = edge[1]
        self.vertices[vname1].removeLink(vname2)
        self.vertices[vname2].removeLink(vname1)
        self.num_edge -= 1
        self.betweenness = None
        self.groups = None
    
    def freshDegrees(self):
        for vname in self.vertices:
            vtx = self.vertices[vname]
            vtx.freshDegree()

    def getDegree(self, vtx_name):
        vtx = self.vertices[vtx_name]
        return vtx.getDegree()

    def getWeight(self, vname1, vname2):
        v1 = self.vertices[vname1]
        if vname2 in v1.links.keys():
            return 1
        else:
            return 0
    
    def __iter__(self):
        return iter(self.vertices.keys())
    
    def bfsGN(self, src):
        """
        BFS for Girvan-Newman Algorithm
        src - vtx_name
        """
        searched = [src] # store current, current's companions, current's parents
        nodes = {src: {'parent': [], 'sp': 1, 'child': []}} # {cur_name: {parent:[parent_name, ...], 'sp': int, child: [child_name, ...]}}
        search_queue = [src]
        while(search_queue != []):
            cnames_found = []
            for cur_name in search_queue:
                cur_sp = nodes[cur_name]['sp']
                cur_vtx = self.vertices[cur_name]
                child_vtx_names = cur_vtx.links.keys()
                for cname in child_vtx_names:
                    if cname not in searched:
                        nodes[cur_name]['child'].append(cname)
                        if cname not in cnames_found:
                            nodes[cname] = {'parent': [cur_name], 'sp': cur_sp, 'child': []}
                            cnames_found.append(cname)
                        else:
                            nodes[cname]['parent'].append(cur_name)
                            nodes[cname]['sp'] += cur_sp
            searched.extend(cnames_found)
            search_queue = cnames_found
        return searched, nodes

    def makeKey(self, str1, str2):
        return tuple(sorted([str1, str2]))
    
    def countEdgesValueOnce(self, src):
        searched, nodes = self.bfsGN(src)
        searched.reverse()
        
        edges_value = {}
        for node in searched:
            cur_node = nodes[node]
            if cur_node.get('parent') == []:
                break
            if cur_node.get('child') == []:
                # a leaf node
                for parent in cur_node['parent']:
                    edge = self.makeKey(node, parent)
                    parent_sp = nodes[parent]['sp']
                    edges_value[edge] = parent_sp / cur_node['sp']
            else:
                # sum edges' value linking children
                value = 1 # self value is 1
                for child in cur_node['child']:
                    edge = self.makeKey(node, child)
                    value += edges_value[edge]
                # calculate edges' value linking parents
                for parent in cur_node['parent']:
                    edge = self.makeKey(node, parent)
                    parent_sp = nodes[parent]['sp']
                    edges_value[edge] = value * parent_sp / cur_node['sp']
        return edges_value
    
    def getBetweenness(self):
        edges_betweenness = {}
        for node in self:
            edges_value = self.countEdgesValueOnce(node)
            for edge in edges_value:
                if edges_betweenness.get(edge) == None:
                    edges_betweenness[edge] = edges_value[edge]
                else:
                    edges_betweenness[edge] += edges_value[edge]
        
        for edge in edges_betweenness:
            edges_betweenness[edge] = edges_betweenness[edge] / 2
        self.betweenness = edges_betweenness
        return edges_betweenness
    
    def _getEdgesWithHighestBetweenness(self):
        if self.betweenness == None:
            return None
        large = -1
        res = set([])
        for pair in self.betweenness:
            b = self.betweenness[pair]
            if b > large:
                res = set([])
                res.add(pair)
                large = b
            elif b == large:
                res.add(pair)
            else:
                continue
        return list(res)

    def removeEdgesWithHighestBetweenness(self):
        if self.betweenness == None:
            self.getBetweenness()
        if self.betweenness == {}:
            # all the nodes are separated
            return None
        edges = self._getEdgesWithHighestBetweenness()
        for pair in edges:
            self.removeEdge(pair)
        return edges

    def bfs(self, src_name):
        searched = [src_name]
        wait = [src_name]
        while(wait != []):
            cur = wait.pop(0)
            cur_vtx = self.vertices[cur]
            children = cur_vtx.links.keys()
            for child in children:
                if child not in searched:
                    searched.append(child)
                    wait.append(child)
        return searched

    def getGroups(self):
        nodes_unsearched = set(self.vertices.keys())
        groups = []
        while(len(nodes_unsearched)):
            src = nodes_unsearched.pop()
            one_group = self.bfs(src)
            for vtx_name in one_group:
                nodes_unsearched.discard(vtx_name)
            groups.append(one_group)
        self.groups = groups
        return groups

    def countModularity(self, groups=None):
        m = self.num_edge
        self.freshDegrees()

        if groups == None:
            groups = self.getGroups()
        
        modularity_sum = 0
        for group in groups:
            sum_group = 0
            for vname1 in group:
                for vname2 in group:
                    Aij = self.getWeight(vname1, vname2)
                    degreei = self.getDegree(vname1)
                    degreej = self.getDegree(vname2)
                    degree_expect = degreei * degreej / (2 * m)
                    sum_group += Aij - degree_expect
            modularity_sum += sum_group
        modularity = modularity_sum / (2 * m)
        return modularity

    def getEdgesShouldBeRemovedBasedOnModularity(self):
        g = UndirectedGraph(vertices=self.vertices_copy, edges=self.edges_copy)

        num_remain_edges = g.num_edge
        res = []
        while(num_remain_edges != 0):
            groups = g.getGroups()
            modularity = self.countModularity(groups)
            removed_edges = g.removeEdgesWithHighestBetweenness()
            num_remain_edges -= len(removed_edges)
            res.append((modularity, removed_edges))

        largest_i = None
        largest_item = res[0]
        for i, item in enumerate(res):
            if item[0] - largest_item[0] > DECIMAL_ACCURACY:
                largest_item = item
                largest_i = i
        
        edges_should_be_removed = [i for x in res[:largest_i] for i in x[1]]

        return edges_should_be_removed

    def getOptimalClustersBasedOnModularity(self):
        edges_should_be_removed = self.getEdgesShouldBeRemovedBasedOnModularity()
        g = UndirectedGraph(vertices=self.vertices_copy, edges=self.edges_copy)

        for e in edges_should_be_removed:
            g.removeEdge(e)
        
        optimal_groups = g.getGroups()
        return optimal_groups










