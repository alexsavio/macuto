
import pandas as pd
from tabulate import tabulate
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

    #https://bitbucket.org/astanin/python-tabulate
    def tabulate(self, tablefmt='plain'):
        """

        :param tablefmt: string
         Supported table formats are:
        "plain"
        "simple"
        "grid"
        "pipe"
        "orgtbl"
        "rst"
        "mediawiki"
        "latex"

        :return: tabulate
        Tabulated content
        """
        return tabulate(self, tablefmt=tablefmt)


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

    #https://bitbucket.org/astanin/python-tabulate
    def tabulate(self, tablefmt='plain'):
        """

        :param tablefmt: string
         Supported table formats are:
        "plain"
        "simple"
        "grid"
        "pipe"
        "orgtbl"
        "rst"
        "mediawiki"
        "latex"

        :return: tabulate
        Tabulated content
        """
        return tabulate(list(self.items()), tablefmt=tablefmt)