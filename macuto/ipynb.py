from collections import OrderedDict

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in
        IPython Notebook. """

    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")

            for col in row:
                html.append("<td>{0}</td>".format(col))

            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


class DictTable(OrderedDict):
    """ Overriden dict class which takes a dictionary
    and renders an HTML Table in IPython Notebook. """

    def _repr_html_(self):
        html = ["<table>"]
        html.append("<tr>")
        keys = self.keys()
        for col in keys:
            html.append("<td><center><b>{0}</b></center></td>".format(col))

        html.append("</tr><tr>")

        for col in keys:
            html.append("<td><center>{0}</center></td>".format(self[col]))

        html.append("</table>")
        return ''.join(html)