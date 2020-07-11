<%
  # Stripped-down and monkey-patched template for pdoc module documentation.
  # This probably only works under very specific circumstances anymore but it
  # should be good enough for a simple, small project.

  import re
  import sys
  import markdown
  import pdoc

  # To display the version number and make lookups in super-packages possible,
  # the root module is needed everywhere
  root_name = module.name.split('.')[0]
  root = pdoc.import_module(root_name)
  root_version = root.__version__
  root_module = pdoc.Module(root)

  # From language reference, but adds '.' to allow fully qualified names.
  pyident = re.compile('^[a-zA-Z_][a-zA-Z0-9_.]+$')

  # Whether we're showing the module list or a single module.
  module_list = 'modules' in context.keys()

  def ident(s):
    return '<span class="ident">%s</span>' % s

  def linkify(match):
    matched = match.group(0)
    ident = matched[1:-1]
    name, url = lookup(ident)
    if name is None:
      return matched
    return '[`%s`](%s)' % (name, url)

  def mark(s, linky=True):
    if linky:
      s, _ = re.subn('\b\n\b', ' ', s)
    if not module_list:
      s, _ = re.subn('`[^`]+`', linkify, s)

    extensions = []
    s = markdown.markdown(s.strip(), extensions=extensions)
    return s

  def glimpse(s, length=100):
    if len(s) < length:
      return s
    return s[0:length] + '...'

  def module_url(m, module=module):
    """Returns a URL for `m`, which must be an instance of `Module`. Also, `m`
    must be a submodule of `module` being documented.

    Namely, '.' import separators are replaced with '/' URL separators. Also,
    packages are translated as directories containing `index.html`
    corresponding to the `__init__` module, while modules are translated as
    regular HTML files with an `.m.html` suffix. (Given default values of
    `pdoc.html_module_suffix` and `pdoc.html_package_name`.)

    Patched with the `module` parameter to enable lookups from the root module
    even in submodules.
    """
    if module.name == m.name:
      return ''

    if len(link_prefix) > 0:
      base = m.name
    else:
      base = m.name[len(module.name)+1:]
    url = base.replace('.', '/')
    if m.is_package():
      url += '/%s' % pdoc.html_package_name
    else:
      url += pdoc.html_module_suffix
    return link_prefix + url

  def external_url(refname):
    """Attempts to guess an absolute URL for the external identifier given.

    Note that this just returns the refname with an ".ext" suffix.  It will be
    up to whatever is interpreting the URLs to map it to an appropriate
    documentation page.
    """
    return '/%s.ext' % refname

  def is_external_linkable(name):
    return external_links and pyident.match(name) and '.' in name

  def lookup(refname, module=module, shorten=False):
    """Given a fully qualified identifier name, return its refname with respect
    to `module` and a value for a `href` attribute. If `refname` is not in the
    public interface of `module` or its submodules, then `None` is returned for
    both return values. (Unless this module has enabled external linking.)

    In particular, this takes into account sub-modules and external
    identifiers. If `refname` is in the public API of `module`, then a local
    anchor link is given. If `refname` is in the public API of a sub-module,
    then a link to a different page with the appropriate anchor is given.
    Otherwise, `refname` is considered external and no link is used.

    Patched with the `module` parameter to enable lookups from the root module
    even in submodules and the `shorten` parameter to sort out attributes of
    the same name on different classes in the same module.
    """
    d = module.find_ident(refname)
    # Nothing found
    if isinstance(d, pdoc.External):
      if is_external_linkable(refname):
        return d.refname, external_url(d.refname)
      # Try to find the name in the root package and its sub-packages. If the
      # name is also not reachable from the root module, there is nothing we
      # can do.
      elif module is not root_module:
        return lookup(refname, module=root_module)
      else:
        return None, None
    # Modules always use the full path
    if isinstance(d, pdoc.Module):
      return d.refname, module_url(d, module=module)
    # Other stuff than methods
    if module.is_public(d.refname):
      # The is-based test sorts out root requests from modules that are not
      # root, where it is necessary to include the full name and the index.html
      # at the beginning of the link
      if module is root_module:
        return d.refname, '%s#%s' % (pdoc.html_package_name, d.refname)
      # Another special case to sort out lookups of attributes with the same
      # name on classes in the current module: keep the class name but strip
      # the name of the current module.
      elif not shorten:
        return d.refname.strip(module.name + "."), '#%s' % d.refname
      else:
        return d.name, '#%s' % d.refname
    return d.refname, '%s#%s' % (module_url(d.module, module=module), d.refname)

  def link(refname, shorten=True):
    """A convenience wrapper around `href` to produce the full `a` tag if
    `refname` is found. Otherwise, plain text of `refname` is returned.

    Use `shorten=True` by default so names in the index menu are shortened.
    """
    name, url = lookup(refname, shorten=shorten)
    if name is None:
      return refname
    return '<a href="%s">%s</a>' % (url, name)

  # pdoc does not generate links to move up the hierarchy again, this should be
  # able to generate links for the (sub-)module path
  def module_nav_links(module=module):
    parts = module.name.split(".")
    links = [_MockModule(".".join(parts[:i+1])).link() for i in range(len(parts))]
    return ".".join(links)

  class _MockModule:
    """A mock module that is used to generate module navigation links"""
    def __init__(self, name):
      self.name = name
      self.is_package = lambda: False
    def link(self):
      url = module_url(self, root_module)
      nam = self.name.split(".")[-1]
      return '<a href="{}">{}</a>'.format(url if url else pdoc.html_package_name, nam)
%>

<%def name="show_desc(d, limit=None)">
  <%
  inherits = (hasattr(d, 'inherits')
           and (len(d.docstring) == 0
            or d.docstring == d.inherits.docstring))
  docstring = (d.inherits.docstring if inherits else d.docstring).strip()
  if limit is not None:
    if docstring:
      docstring = docstring.splitlines()[0]
    docstring = glimpse(docstring, limit)
  %>
  % if len(docstring) > 0:
  % if inherits:
    <div class="desc inherited">${docstring | mark}</div>
  % else:
    <div class="desc">${docstring | mark}</div>
  % endif
  % endif
</%def>

<%def name="show_inheritance(d)">
  % if hasattr(d, 'inherits'):
    <p class="inheritance">
     <strong>Inheritance:</strong>
     % if hasattr(d.inherits, 'cls'):
       <code>${link(d.inherits.cls.refname)}</code>.<code>${link(d.inherits.refname)}</code>
     % else:
       <code>${link(d.inherits.refname)}</code>
     % endif
    </p>
  % endif
</%def>

<%def name="show_column_list(items, numcols=3)">
  <ul>
  % for item in items:
    <li>${item}</li>
  % endfor
  </ul>
</%def>

<%def name="show_module(module)">
  <%
  variables = module.variables()
  classes = module.classes()
  functions = module.functions()
  submodules = module.submodules()
  %>

  <%def name="show_method(c, f)">
  <div class="item">
    <div class="name def" id="${f.refname}">
      ${c.name}.${ident(f.name)}(${f.spec() | h})
    </div>
    ${show_desc(f)}
  </div>
  </%def>

  <header id="section-intro">
  <h1 class="title"><span class="name">${module.name}</span></h1>
  ${module.docstring | mark}
  </header>

  <section id="section-items">

    % if len(submodules) > 0:
    <h2 class="section-title" id="header-submodules">Sub-modules</h2>
    % for m in submodules:
      <div class="item">
      <p class="name">${link(m.refname)}</p>
      ${show_desc(m, limit=300)}
      </div>
    % endfor
    % endif

    % if len(classes) > 0:
    <h2 class="section-title" id="header-classes">Classes</h2>
    % for c in classes:
      <%
      class_vars = c.class_variables()
      inst_vars = c.instance_variables()
      mro = c.module.mro(c)
      # For some reason, methods and staticmethods were swapped
      methods = c.functions()
      smethods = c.methods()
      %>
      <div class="item">
      <p id="${c.refname}" class="name class-name">class ${ident(c.name)}</p>
      ${show_desc(c)}

      <div class="class">
        % if len(mro) > 2:
          <h3>Ancestors</h3>
          <ul class="class_list">
          % for cls in mro[1:-1]: ## Hide class itself and builtins.object
          <li>${link(cls.refname)}</li>
          % endfor
          </ul>
        % endif
        % if len(class_vars) > 0:
          <h3>Class attributes</h3>
          % for v in class_vars:
            <div class="item">
            <p id="${v.refname}" class="name">${ident(v.name)}</p>
            ${show_inheritance(v)}
            ${show_desc(v)}
            </div>
          % endfor
        % endif
        % if len(smethods) > 0:
          <h3>Static Methods</h3>
          % for f in smethods:
            ${show_method(c, f)}
          % endfor
        % endif
        % if len(methods) > 0:
          <h3>Methods</h3>
          % for f in methods:
            ${show_method(c, f)}
          % endfor
        % endif
        % if len(inst_vars) > 0:
          <h3>Instance Attributes</h3>
          % for v in inst_vars:
            <div class="item">
            <p id="${v.refname}" class="name">${ident(c.name)}.${ident(v.name)}</p>
            ${show_inheritance(v)}
            ${show_desc(v)}
            </div>
          % endfor
        % endif
      </div>
      </div>
    % endfor
    % endif

    % if len(functions) > 0:
    <h2 class="section-title" id="header-functions">Functions</h2>

    % for f in functions:
      <div class="item">
        <div class="name def" id="${f.refname}">
        ${ident(f.name)}(${f.spec() | h})
        </div>
        ${show_desc(f)}
      </div>
    % endfor
    % endif

    % if len(variables) > 0:
    <h2 class="section-title" id="header-variables">Module variables</h2>
    % for v in variables:
      <div class="item">
      <p id="${v.refname}" class="name">${ident(v.name)}</p>
      ${show_desc(v)}
      </div>
    % endfor
    % endif

  </section>
</%def>

<%def name="module_index(module)">
  <%
  variables = module.variables()
  classes = module.classes()
  functions = module.functions()
  submodules = module.submodules()
  %>
  <nav id="sidebar">
    <h1>${root_name} ${root_version}</h1>
    <hr>
    <p>${module_nav_links()}</p>
    <hr>
    <ul id="index">

    % if len(submodules) > 0:
    <li class="set"><h3><a href="#header-submodules">Sub-modules</a></h3>
      <ul>
      % for m in submodules:
        <li>${link(m.refname)}</li>
      % endfor
      </ul>
    </li>
    % endif

    % if len(classes) > 0:
    <li class="set"><h3><a href="#header-classes">Classes</a></h3>
      <ul>
      % for c in classes:
        <li>
        ${link(c.refname)}
        <%
          methods = c.methods() + c.functions() + c.instance_variables() + c.class_variables()
        %>
        % if len(methods) > 0:
          ${show_column_list(sorted(map(lambda f: link(f.refname), methods)))}
        % endif
        </li>
      % endfor
      </ul>
    </li>
    % endif

    % if len(functions) > 0:
    <li class="set"><h3><a href="#header-functions">Functions</a></h3>
      ${show_column_list(map(lambda f: link(f.refname), functions))}
    </li>
    % endif

    % if len(variables) > 0:
    <li class="set"><h3><a href="#header-variables">Module variables</a></h3>
      ${show_column_list(map(lambda v: link(v.refname), variables))}
    </li>
    % endif

    </ul>
  </nav>
</%def>

<!doctype html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />
<title>${module.name} | ${root_name} ${root_version} documentation</title>
<style type="text/css">
<%include file="css.mako"/>
</style>
</head>
<body>
<div id="container">
${module_index(module)}
<main id="content">
${show_module(module)}
</main>
</div>
</body>
</html>

