<%
color1 = "#039"
color2 = "#09f"
%>
html, body {
  margin: 0;
  padding: 0;
  min-height: 100%;
}
body {
  background-color: #eee;
  font-family: Helvetica, sans;
  font-size: 11pt;
  line-height: 1.5em;
}

#container {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap-reverse;
  height: 100%;
}
main {
  flex: 1 1;
  max-width: 750px;
  padding: 15px;
  background-color: #fff;
  border-left: 2px solid #ccc;
}
nav {
  flex: 0 1;
  min-width: 250px;
  padding: 15px;
  overflow: hidden;
}

h1, h2, h3, h4, h5 {
  font-weight: 300;
}
h1 {
  font-size: 2.2em;
  line-height: 1.1em;
  margin: 0 0 0.8em 0;
}
h2 {
  font-size: 1.6em;
  margin: 1.8em 0 .8em 0;
}
h3 {
  font-size: 1.4em;
  margin: 20px 0 15px 0;
}

a {
  color: ${color1};
  text-decoration: none;
}
a:hover {
  color: ${color2};
}

pre, code, .mono, .name {
  font-family: monospace;
  font-size: 0.95em;
}
code {
    display: inline-block;
    line-height: 1.2em;
}

ul {
  padding-left: 2em;
  list-style-type: disc;
}

hr {
  border: 1px solid #ccc;
}

.title .name {
  font-weight: bold;
}

.section-title {
  margin-top: 2em;
}

.ident {
  font-weight: bold;
}

pre {
  border-left: 2px solid #ccc;
  padding: 5px 10px;
}

nav a {
  color: #000;
}
nav a:hover {
  color: ${color1};
}
nav h1 {
  font-size: 1.8em;
  color: ${color1};
}
#index {
  list-style-type: none;
  margin: 0;
  padding: 0;
  font-size: 0.9em;
  line-height: 1.4em;
}
#index ul ul {
  list-style-type: circle;
}

.item {
  margin: 0 0 25px 0;
}
.item .class {
  margin: 0 0 25px 30px;
}
.item .class ul.class_list {
  margin: 0 0 20px 0;
}
.item .name {
  background: #eee;
  margin: 0 0 5px 0;
  padding: 5px 10px 3px 10px;
  display: inline-block;
  min-width: 40%;
  border-bottom: 2px solid #ccc;
}
.item .def {
  padding-left: 40px;
  text-indent: -30px;
}
.item .class-name {
  display: block;
}
.item .inheritance {
  margin: 3px 0 0 25px;
}
.item .inherited {
  color: #666;
}
.item .desc {
  padding: 0 8px;
  margin: 3px 0;
}
.item .desc p, .item .desc ul, .item .desc pre {
  margin: 0 0 8px 0;
}

.desc h1, .desc h2, .desc h3 {
  font-size: 100% !important;
}

